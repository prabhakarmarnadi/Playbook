"""
Demo data preparation script — generates synthetic agreements + pre-caches LLM responses.

Run the night before the hackathon:
    python scripts/prep_demo_data.py

Legacy equivalent: Manual CUAD dataset prep + AIDB data loading.
V2: Self-contained data generation + pipeline caching.
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SAMPLE_DIR

# ── Synthetic agreement templates ──────────────────────────────────────────────
TEMPLATES = {
    "saas": {
        "count": 10,
        "title_prefix": "SaaS Agreement",
        "parties": [
            ("Acme Software Inc.", "Widget Corp."),
            ("CloudServe Ltd.", "Digital Solutions LLC"),
            ("NetApps Inc.", "Enterprise Systems Group"),
            ("DataFlow Platform Inc.", "Global Manufacturing Co."),
            ("SaaSify Technologies", "RetailMax Holdings"),
        ],
        "clauses": {
            "payment_terms": [
                "Payment Terms. Licensee shall pay all fees within {days} days of invoice date. Payment shall be made by {method}. Late payments shall accrue interest at a rate of {late_fee}% per month. All fees are in {currency}.",
                "Fees and Payment. The annual subscription fee is ${price:,} per user for {quantity} users. Payment is due net {days}. The total contract value for the initial {term}-year term is ${tcv:,}.",
            ],
            "liability": [
                "Limitation of Liability. IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY EXCEED {cap_type} {cap_amount}. Neither party shall be liable for any indirect, incidental, special, consequential, or punitive damages.",
                "Liability Cap. The total liability of the Provider under this Agreement shall not exceed {cap_amount}, representing {cap_type} the fees paid during the preceding twelve (12) month period.",
            ],
            "auto_renewal": [
                "Term and Renewal. This Agreement shall have an initial term of {term} year(s) commencing on the Effective Date. The Agreement shall automatically renew for successive {renewal_term}-year periods unless either party provides written notice of non-renewal at least {notice_period} days prior to the end of the then-current term.",
            ],
            "data_processing": [
                "Data Processing. Provider shall process Customer Data solely for the purpose of providing the Services. Provider shall maintain appropriate technical and organizational measures to protect Customer Data. Provider shall comply with {regulation} and shall notify Customer of any data breach within {breach_notice} hours.",
            ],
            "sla": [
                "Service Level Agreement. Provider guarantees {uptime}% monthly uptime for the platform. Scheduled maintenance windows are excluded. If uptime falls below the guaranteed level, Customer shall receive service credits equal to {credit_pct}% of monthly fees per {credit_unit}% of downtime below the threshold.",
            ],
        },
    },
    "nda": {
        "count": 5,
        "title_prefix": "Non-Disclosure Agreement",
        "parties": [
            ("TechVenture Inc.", "StartupAI Corp."),
            ("BioResearch Labs", "PharmaCo International"),
            ("FinServe Holdings", "DataAnalytics Ltd."),
        ],
        "clauses": {
            "confidentiality": [
                "Confidentiality Obligations. The Receiving Party shall hold all Confidential Information in strict confidence and shall not disclose such information to any third party without the prior written consent of the Disclosing Party. Confidential Information includes, but is not limited to, trade secrets, business plans, financial data, customer lists, and technical specifications.",
            ],
            "term_survival": [
                "Term and Survival. This Agreement shall remain in effect for {term} years from the Effective Date. The confidentiality obligations shall survive termination for an additional {survival} years. Upon termination, the Receiving Party shall return or destroy all Confidential Information within {return_days} days.",
            ],
        },
    },
    "vendor": {
        "count": 5,
        "title_prefix": "Vendor Agreement",
        "parties": [
            ("SupplyChain Solutions", "MegaRetail Inc."),
            ("Industrial Components Ltd.", "AutoManufacture Co."),
        ],
        "clauses": {
            "liability": [
                "Indemnification and Liability. Vendor shall indemnify and hold harmless Company from any claims arising from Vendor's breach of this Agreement. Vendor's total liability under this Agreement shall not exceed ${cap:,}. The liability cap does not apply to breaches of confidentiality or willful misconduct.",
            ],
            "pricing": [
                "Pricing and Payment. The unit price for Products is ${unit_price:.2f} per unit. Company commits to purchasing a minimum of {min_quantity:,} units per quarter. Volume discounts apply: {discount_pct}% discount for orders exceeding {discount_threshold:,} units.",
            ],
        },
    },
    "employment": {
        "count": 5,
        "title_prefix": "Employment Agreement",
        "parties": [
            ("TechCorp Global", ""),
            ("FinanceFirst Ltd.", ""),
        ],
        "clauses": {
            "non_compete": [
                "Non-Competition. Employee agrees not to engage in any business that competes with the Company for a period of {non_compete_months} months following termination of employment, within a {radius_miles}-mile radius of any Company office. This restriction applies to the {industry} industry.",
            ],
            "compensation": [
                "Compensation. Employee shall receive an annual base salary of ${salary:,}. Employee is eligible for an annual performance bonus of up to {bonus_pct}% of base salary. Equity compensation includes {equity_shares:,} stock options vesting over {vesting_years} years with a one-year cliff.",
            ],
        },
    },
    "lease": {
        "count": 5,
        "title_prefix": "Commercial Lease Agreement",
        "parties": [
            ("PropertyCo Holdings", "TechStartup Inc."),
            ("RealEstate Ventures", "RetailChain Corp."),
        ],
        "clauses": {
            "rent": [
                "Rent. Tenant shall pay monthly rent of ${monthly_rent:,} due on the first day of each month. Rent shall increase by {escalation_pct}% per annum. Late payment fee: ${late_fee} per day. Security deposit: ${deposit:,}.",
            ],
            "term": [
                "Lease Term. The initial lease term is {term_years} years commencing on {start_date}. Tenant has the option to renew for an additional {renewal_years} year(s) by providing {notice_days} days written notice prior to expiration.",
            ],
        },
    },
}

# Randomization parameters
import random
random.seed(42)


def generate_agreement(category: str, template: dict, idx: int) -> dict:
    """Generate a single synthetic agreement."""
    parties = random.choice(template["parties"])
    party_a = parties[0]
    party_b = parties[1] if parties[1] else f"Employee #{idx+1}"

    sections = []
    sections.append(f"# {template['title_prefix']}\n")
    sections.append(f"This {template['title_prefix']} (the \"Agreement\") is entered into as of "
                     f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d} "
                     f"by and between {party_a} (\"{party_a.split()[0]}\") and {party_b}.\n")

    for clause_type, clause_templates in template["clauses"].items():
        clause_text = random.choice(clause_templates)

        # Fill in template variables
        clause_text = clause_text.format(
            days=random.choice([30, 45, 60]),
            method=random.choice(["wire transfer", "ACH", "check"]),
            late_fee=random.choice([1.0, 1.5, 2.0]),
            currency=random.choice(["USD", "EUR", "GBP"]),
            price=random.choice([500, 1000, 2500, 5000]),
            quantity=random.choice([50, 100, 250, 500]),
            term=random.choice([1, 2, 3]),
            tcv=random.choice([150000, 500000, 1500000, 4500000]),
            cap_type=random.choice(["the total fees paid under", "two times"]),
            cap_amount=random.choice(["$1,000,000", "$5,000,000", "$10,000,000"]),
            renewal_term=random.choice([1, 2]),
            notice_period=random.choice([30, 60, 90]),
            regulation=random.choice(["GDPR", "CCPA", "SOC 2"]),
            breach_notice=random.choice([24, 48, 72]),
            uptime=random.choice([99.5, 99.9, 99.95]),
            credit_pct=random.choice([5, 10, 25]),
            credit_unit=random.choice([0.1, 0.5, 1.0]),
            survival=random.choice([2, 3, 5]),
            return_days=random.choice([10, 30, 45]),
            cap=random.choice([1000000, 5000000, 10000000]),
            unit_price=random.uniform(10, 500),
            min_quantity=random.choice([1000, 5000, 10000]),
            discount_pct=random.choice([5, 10, 15]),
            discount_threshold=random.choice([10000, 25000, 50000]),
            non_compete_months=random.choice([6, 12, 18, 24]),
            radius_miles=random.choice([25, 50, 100]),
            industry=random.choice(["software development", "financial technology", "healthcare IT"]),
            salary=random.choice([120000, 150000, 200000, 250000]),
            bonus_pct=random.choice([10, 15, 20, 25]),
            equity_shares=random.choice([10000, 25000, 50000, 100000]),
            vesting_years=random.choice([3, 4]),
            monthly_rent=random.choice([5000, 10000, 25000, 50000]),
            escalation_pct=random.choice([2, 3, 5]),
            deposit=random.choice([10000, 25000, 50000]),
            term_years=random.choice([3, 5, 10]),
            start_date=f"2024-{random.randint(1,12):02d}-01",
            renewal_years=random.choice([2, 3, 5]),
            notice_days=random.choice([60, 90, 120, 180]),
        )
        sections.append(f"\n## {clause_type.replace('_', ' ').title()}\n\n{clause_text}\n")

    return {
        "id": f"AGR_{category.upper()}_{idx+1:03d}",
        "type": template["title_prefix"],
        "text": "\n".join(sections),
        "metadata": {
            "category": category,
            "party_a": party_a,
            "party_b": party_b,
            "date_created": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        },
    }


def main():
    output_dir = SAMPLE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_docs = []
    for category, template in TEMPLATES.items():
        for i in range(template["count"]):
            doc = generate_agreement(category, template, i)
            all_docs.append(doc)

            # Write individual .txt file
            txt_path = output_dir / f"{doc['id']}.txt"
            txt_path.write_text(doc["text"])

    # Write combined JSON (compatible with legacy sample_documents.json format)
    json_path = output_dir / "sample_documents.json"
    json_data = {"documents": all_docs}
    json_path.write_text(json.dumps(json_data, indent=2))

    print(f"Generated {len(all_docs)} synthetic agreements:")
    for category, template in TEMPLATES.items():
        print(f"  {category}: {template['count']} docs")
    print(f"\nOutput: {output_dir}")
    print(f"  - {len(all_docs)} .txt files")
    print(f"  - 1 sample_documents.json")


if __name__ == "__main__":
    main()
