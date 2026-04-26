"""
Field Value Normalization
=========================
Takes 23,225+ raw text extractions and normalizes them into structured types:
  - Monetary amounts  → {"amount": 500000, "currency": "USD", "raw": "$500,000"}
  - Durations         → {"value": 30, "unit": "days", "raw": "thirty (30) days"}
  - Dates             → {"date": "2026-01-01", "format": "iso", "raw": "January 1, 2026"}
  - Percentages       → {"value": 5.5, "raw": "5.5%"}
  - Booleans          → {"value": true, "raw": "mutual"}
  - Enumerations      → {"value": "delaware", "category": "jurisdiction", "raw": "State of Delaware"}
  - Free text         → {"value": "...", "type": "text"} (no normalization)

This enables portfolio-wide aggregation (sum of TCV, avg notice period, etc.).
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

# ── Monetary patterns ──
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
_CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL"}

_MONEY_RE = re.compile(
    r'(?P<sym>[€£¥₹$])\s*(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<scale>million|billion|thousand|M|B|K)?'
    r'|(?P<amount2>[\d,]+(?:\.\d+)?)\s*(?P<scale2>million|billion|thousand|M|B|K)?\s*(?P<code>[A-Z]{3})'
    r'|(?P<code2>[A-Z]{3})\s*(?P<amount3>[\d,]+(?:\.\d+)?)\s*(?P<scale3>million|billion|thousand|M|B|K)?',
    re.IGNORECASE,
)

_SCALE_MAP = {
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
    "thousand": 1_000, "k": 1_000,
}

# ── Duration patterns ──
_DURATION_RE = re.compile(
    r'(?P<num>\d+)\s*(?:\(?(?P<num2>\d+)\)?\s*)?'
    r'(?P<unit>day|week|month|year|business day|calendar day|working day)s?',
    re.IGNORECASE,
)

_WRITTEN_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "sixty": 60,
    "ninety": 90, "one hundred": 100, "hundred": 100,
    "one hundred eighty": 180, "three hundred sixty-five": 365,
    "twenty-four": 24, "forty-eight": 48, "seventy-two": 72,
}

_WRITTEN_DURATION_RE = re.compile(
    r'(?P<word>' + '|'.join(re.escape(w) for w in sorted(_WRITTEN_NUMBERS.keys(), key=len, reverse=True)) +
    r')\s*\(?(?P<num>\d+)?\)?\s*(?P<unit>day|week|month|year|business day|calendar day|working day)s?',
    re.IGNORECASE,
)

# ── Date patterns ──
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_PATTERNS = [
    # January 1, 2026 or Jan 1, 2026
    re.compile(r'(?P<month>' + '|'.join(_MONTHS.keys()) + r')\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})', re.IGNORECASE),
    # 2026-01-01 or 2026/01/01
    re.compile(r'(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})'),
    # 01/01/2026 or 01-01-2026 (MM/DD/YYYY)
    re.compile(r'(?P<month>\d{1,2})[-/](?P<day>\d{1,2})[-/](?P<year>\d{4})'),
    # 1st day of January, 2026
    re.compile(r'(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?P<month>' +
               '|'.join(_MONTHS.keys()) + r'),?\s+(?P<year>\d{4})', re.IGNORECASE),
]

# ── Percentage patterns ──
_PCT_RE = re.compile(r'(?P<num>\d+(?:\.\d+)?)\s*%|(?P<num2>\d+(?:\.\d+)?)\s*percent', re.IGNORECASE)

# ── Boolean detection ──
_TRUE_WORDS = {"yes", "true", "mutual", "applicable", "included", "required", "agreed"}
_FALSE_WORDS = {"no", "false", "n/a", "not applicable", "excluded", "none", "waived"}


@dataclass
class NormalizedValue:
    raw: str
    norm_type: str  # monetary, duration, date, percentage, boolean, enum, text
    structured: dict


class FieldNormalizer:
    """Normalizes raw extraction values into structured types."""

    def __init__(self, db_path: str):
        self.db = duckdb.connect(db_path, read_only=False)
        self._ensure_tables()

    def _ensure_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS normalized_values (
                extraction_id VARCHAR PRIMARY KEY,
                field_id VARCHAR,
                agreement_id VARCHAR,
                raw_value VARCHAR,
                norm_type VARCHAR,
                structured JSON,
                normalized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def normalize_all(self) -> dict:
        """Normalize all extractions."""
        rows = self.db.execute("""
            SELECT e.extraction_id, e.field_id, e.agreement_id, e.value,
                   fd.name as field_name, fd.field_type
            FROM extractions e
            JOIN field_definitions fd ON e.field_id = fd.field_id
            WHERE e.value IS NOT NULL AND e.value != ''
        """).fetchall()

        logger.info(f"Normalizing {len(rows)} extractions...")

        counts = {"monetary": 0, "duration": 0, "date": 0, "percentage": 0,
                  "boolean": 0, "enum": 0, "text": 0}
        batch = []

        for ext_id, field_id, agr_id, value, field_name, field_type in rows:
            norm = self._normalize(value, field_name, field_type)
            counts[norm.norm_type] += 1

            batch.append((ext_id, field_id, agr_id, value,
                         norm.norm_type, json.dumps(norm.structured)))

            if len(batch) >= 1000:
                self._flush_batch(batch)
                batch = []

        if batch:
            self._flush_batch(batch)

        logger.info(f"Normalization complete: {counts}")
        return counts

    def _flush_batch(self, batch: list):
        self.db.executemany("""
            INSERT OR REPLACE INTO normalized_values
            (extraction_id, field_id, agreement_id, raw_value, norm_type, structured)
            VALUES (?, ?, ?, ?, ?, ?)
        """, batch)

    def _normalize(self, value: str, field_name: str, field_type: str) -> NormalizedValue:
        """Determine the best normalization for a value."""
        if not value or not value.strip():
            return NormalizedValue(raw=value, norm_type="text", structured={"value": value})

        val_lower = value.strip().lower()
        fname_lower = (field_name or "").lower()

        # 1. Try monetary (by field name hint or content)
        if self._is_monetary_field(fname_lower) or self._has_currency_signal(value):
            result = self._parse_monetary(value)
            if result:
                return NormalizedValue(raw=value, norm_type="monetary", structured=result)

        # 2. Try date (by field name hint or content)
        if self._is_date_field(fname_lower):
            result = self._parse_date(value)
            if result:
                return NormalizedValue(raw=value, norm_type="date", structured=result)

        # 3. Try duration (by field name hint or content)
        if self._is_duration_field(fname_lower):
            result = self._parse_duration(value)
            if result:
                return NormalizedValue(raw=value, norm_type="duration", structured=result)

        # 4. Try percentage
        if self._is_pct_field(fname_lower) or "%" in value:
            result = self._parse_percentage(value)
            if result:
                return NormalizedValue(raw=value, norm_type="percentage", structured=result)

        # 5. Try boolean
        if val_lower in _TRUE_WORDS or val_lower in _FALSE_WORDS:
            return NormalizedValue(raw=value, norm_type="boolean",
                                  structured={"value": val_lower in _TRUE_WORDS, "raw": value})

        # 6. Content-based fallback (no field name hint)
        result = self._parse_date(value)
        if result:
            return NormalizedValue(raw=value, norm_type="date", structured=result)

        result = self._parse_monetary(value)
        if result:
            return NormalizedValue(raw=value, norm_type="monetary", structured=result)

        result = self._parse_duration(value)
        if result:
            return NormalizedValue(raw=value, norm_type="duration", structured=result)

        # 7. Short values → enum candidate
        if len(value.strip()) < 60 and "\n" not in value:
            return NormalizedValue(raw=value, norm_type="enum",
                                  structured={"value": value.strip(), "category": fname_lower})

        # 8. Free text
        return NormalizedValue(raw=value, norm_type="text", structured={"value": value})

    # ── Field name heuristics ──

    def _is_monetary_field(self, name: str) -> bool:
        return any(kw in name for kw in [
            "amount", "price", "fee", "cost", "payment", "cap", "limit",
            "penalty", "compensation", "salary", "rent", "premium",
            "deductible", "threshold", "value", "tcv", "acv", "arv",
        ])

    def _is_date_field(self, name: str) -> bool:
        return any(kw in name for kw in [
            "date", "effective", "expiration", "expiry", "commence",
            "start", "end", "termination_date", "renewal_date", "signed",
        ])

    def _is_duration_field(self, name: str) -> bool:
        return any(kw in name for kw in [
            "period", "term", "duration", "notice", "cure", "days",
            "months", "years", "timeline", "window", "deadline",
        ])

    def _is_pct_field(self, name: str) -> bool:
        return any(kw in name for kw in [
            "percent", "rate", "ratio", "margin", "discount",
            "interest", "escalation", "increase",
        ])

    def _has_currency_signal(self, value: str) -> bool:
        return any(sym in value for sym in _CURRENCY_SYMBOLS) or \
               bool(re.search(r'\b[A-Z]{3}\s*[\d,]+', value))

    # ── Parsers ──

    def _parse_monetary(self, value: str) -> Optional[dict]:
        match = _MONEY_RE.search(value)
        if not match:
            return None

        # Extract amount
        amount_str = match.group("amount") or match.group("amount2") or match.group("amount3")
        if not amount_str:
            return None

        cleaned = amount_str.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            amount = float(cleaned)
        except ValueError:
            return None

        # Scale
        scale_str = (match.group("scale") or match.group("scale2") or match.group("scale3") or "").lower()
        if scale_str in _SCALE_MAP:
            amount *= _SCALE_MAP[scale_str]

        # Currency
        sym = match.group("sym")
        code = match.group("code") or match.group("code2")
        if sym:
            currency = _CURRENCY_SYMBOLS.get(sym, "USD")
        elif code and code.upper() in _CURRENCY_CODES:
            currency = code.upper()
        else:
            currency = "USD"

        return {"amount": amount, "currency": currency, "raw": value.strip()}

    def _parse_duration(self, value: str) -> Optional[dict]:
        # Try written number form first: "thirty (30) days"
        match = _WRITTEN_DURATION_RE.search(value)
        if match:
            word = match.group("word").lower()
            num = _WRITTEN_NUMBERS.get(word)
            if match.group("num"):
                num = int(match.group("num"))
            unit = match.group("unit").lower()
            if num:
                return {"value": num, "unit": unit, "raw": value.strip()}

        # Numeric form: "30 days"
        match = _DURATION_RE.search(value)
        if match:
            num = int(match.group("num2") or match.group("num"))
            unit = match.group("unit").lower()
            return {"value": num, "unit": unit, "raw": value.strip()}

        return None

    def _parse_date(self, value: str) -> Optional[dict]:
        for pattern in _DATE_PATTERNS:
            match = pattern.search(value)
            if not match:
                continue

            groups = match.groupdict()
            month_raw = groups.get("month", "")
            day = int(groups.get("day", 0))
            year = int(groups.get("year", 0))

            if month_raw.isdigit():
                month = int(month_raw)
            else:
                month = _MONTHS.get(month_raw.lower(), 0)

            if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                iso = f"{year:04d}-{month:02d}-{day:02d}"
                return {"date": iso, "format": "iso", "raw": value.strip()}

        return None

    def _parse_percentage(self, value: str) -> Optional[dict]:
        match = _PCT_RE.search(value)
        if match:
            num = float(match.group("num") or match.group("num2"))
            return {"value": num, "raw": value.strip()}
        return None

    # ── Aggregation queries ──

    def get_monetary_summary(self) -> list[dict]:
        """Aggregate monetary values across the portfolio."""
        rows = self.db.execute("""
            SELECT fd.name, COUNT(*) as count,
                   MIN(CAST(json_extract(nv.structured, '$.amount') AS DOUBLE)) as min_amount,
                   MAX(CAST(json_extract(nv.structured, '$.amount') AS DOUBLE)) as max_amount,
                   AVG(CAST(json_extract(nv.structured, '$.amount') AS DOUBLE)) as avg_amount
            FROM normalized_values nv
            JOIN field_definitions fd ON nv.field_id = fd.field_id
            WHERE nv.norm_type = 'monetary'
            GROUP BY fd.name
            HAVING COUNT(*) >= 3
            ORDER BY avg_amount DESC
        """).fetchall()
        return [{"field": r[0], "count": r[1], "min": r[2], "max": r[3], "avg": round(r[4], 2)}
                for r in rows]

    def get_duration_summary(self) -> list[dict]:
        """Aggregate duration values across the portfolio."""
        rows = self.db.execute("""
            SELECT fd.name,
                   json_extract_string(nv.structured, '$.unit') as unit,
                   COUNT(*) as count,
                   MIN(CAST(json_extract(nv.structured, '$.value') AS INTEGER)) as min_val,
                   MAX(CAST(json_extract(nv.structured, '$.value') AS INTEGER)) as max_val,
                   AVG(CAST(json_extract(nv.structured, '$.value') AS DOUBLE)) as avg_val
            FROM normalized_values nv
            JOIN field_definitions fd ON nv.field_id = fd.field_id
            WHERE nv.norm_type = 'duration'
            GROUP BY fd.name, unit
            HAVING COUNT(*) >= 3
            ORDER BY count DESC
        """).fetchall()
        return [{"field": r[0], "unit": r[1], "count": r[2],
                 "min": r[3], "max": r[4], "avg": round(r[5], 1)}
                for r in rows]

    def get_date_range(self) -> dict:
        """Get earliest and latest dates in the portfolio."""
        rows = self.db.execute("""
            SELECT MIN(json_extract_string(structured, '$.date')) as earliest,
                   MAX(json_extract_string(structured, '$.date')) as latest,
                   COUNT(*) as count
            FROM normalized_values
            WHERE norm_type = 'date'
        """).fetchone()
        return {"earliest": rows[0], "latest": rows[1], "count": rows[2]}

    def summary(self) -> str:
        """Print normalization summary."""
        counts = self.db.execute("""
            SELECT norm_type, COUNT(*) FROM normalized_values
            GROUP BY norm_type ORDER BY COUNT(*) DESC
        """).fetchall()

        if not counts:
            return "No normalizations computed yet."

        total = sum(c[1] for c in counts)
        lines = [f"Field Normalization Summary ({total} values)"]
        lines.append("─" * 50)
        for norm_type, count in counts:
            pct = 100 * count / total
            bar = "█" * int(pct / 2)
            lines.append(f"  {norm_type:<12} {count:>8} ({pct:>5.1f}%) {bar}")

        # Monetary highlights
        money = self.get_monetary_summary()
        if money:
            lines.append("")
            lines.append("Top monetary fields:")
            for m in money[:5]:
                lines.append(f"  {m['field']}: ${m['min']:,.0f} – ${m['max']:,.0f} "
                           f"(avg ${m['avg']:,.0f}, n={m['count']})")

        # Duration highlights
        durations = self.get_duration_summary()
        if durations:
            lines.append("")
            lines.append("Top duration fields:")
            for d in durations[:5]:
                lines.append(f"  {d['field']}: {d['min']}–{d['max']} {d['unit']} "
                           f"(avg {d['avg']:.0f}, n={d['count']})")

        return "\n".join(lines)

    def close(self):
        self.db.close()
