"""
PDF parsing pipeline — batch-convert PDFs to text and store locally.

Uses PyMuPDF (fitz) for extraction with page-level metadata.
Outputs one .txt file per PDF into the configured output directory,
plus a manifest JSON for downstream pipeline consumption.

Usage:
    poetry run python -m core.pdf_parser /path/to/pdfs
    poetry run python -m core.pdf_parser /path/to/pdfs --output /path/to/output
"""
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import fitz  # pymupdf

from config import DATA_DIR

logger = logging.getLogger(__name__)

PARSED_TEXT_DIR = DATA_DIR / "parsed_texts"


def parse_pdf(pdf_path: str | Path) -> dict:
    """
    Extract text and metadata from a single PDF using PyMuPDF.

    Returns:
        {
            "agreement_id": str,
            "filename": str,
            "raw_text": str,
            "page_count": int,
            "char_count": int,
            "pages": [{"page_num": int, "text": str, "char_count": int}, ...],
            "metadata": {title, author, subject, ...} from PDF info dict,
        }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    pages = []
    full_text_parts = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        pages.append({
            "page_num": page_num + 1,
            "text": page_text,
            "char_count": len(page_text),
        })
        full_text_parts.append(page_text)

    raw_text = "\n".join(full_text_parts)

    # Extract PDF metadata (title, author, etc.)
    pdf_meta = doc.metadata or {}
    doc.close()

    return {
        "agreement_id": f"agr_{uuid.uuid4().hex[:8]}",
        "filename": pdf_path.name,
        "raw_text": raw_text,
        "page_count": len(pages),
        "char_count": len(raw_text),
        "pages": pages,
        "metadata": {
            "title": pdf_meta.get("title", ""),
            "author": pdf_meta.get("author", ""),
            "subject": pdf_meta.get("subject", ""),
            "creator": pdf_meta.get("creator", ""),
            "creation_date": pdf_meta.get("creationDate", ""),
        },
    }


def parse_pdf_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """
    Parse all PDFs in a directory, save extracted text as .txt files,
    and write a manifest JSON.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Where to write .txt files + manifest.json.
                    Defaults to data/parsed_texts/.

    Returns:
        List of parsed document dicts (same shape as pipeline agreements).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else PARSED_TEXT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.rglob("*.pdf"))
    txt_files = sorted(input_dir.rglob("*.txt"))
    all_files = pdf_files + txt_files
    if not all_files:
        logger.warning(f"No PDF or TXT files found in {input_dir}")
        return []

    logger.info(f"Parsing {len(pdf_files)} PDFs + {len(txt_files)} TXTs from {input_dir}")

    results = []
    errors = []

    for file_path in all_files:
        try:
            if file_path.suffix.lower() == ".pdf":
                parsed = parse_pdf(file_path)
            else:
                # Direct TXT ingestion — no PyMuPDF needed
                raw_text = file_path.read_text(encoding="utf-8", errors="replace")
                parsed = {
                    "agreement_id": f"agr_{uuid.uuid4().hex[:8]}",
                    "filename": file_path.name,
                    "raw_text": raw_text,
                    "page_count": 1,
                    "char_count": len(raw_text),
                    "pages": [{"page_num": 1, "text": raw_text, "char_count": len(raw_text)}],
                    "metadata": {"title": file_path.stem, "author": "", "subject": "",
                                 "creator": "", "creation_date": ""},
                }

            # Write extracted text to .txt file in output dir
            txt_name = file_path.stem + ".txt"
            txt_path = output_dir / txt_name
            txt_path.write_text(parsed["raw_text"], encoding="utf-8")

            parsed["txt_path"] = str(txt_path)
            results.append(parsed)

            logger.info(
                f"  {file_path.name}: {parsed['page_count']} pages, "
                f"{parsed['char_count']:,} chars → {txt_name}"
            )
        except Exception as e:
            logger.error(f"  Failed to parse {file_path.name}: {e}")
            errors.append({"filename": file_path.name, "error": str(e)})

    # Write manifest
    manifest = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "parsed_at": datetime.now().isoformat(),
        "total_files": len(all_files),
        "total_pdfs": len(pdf_files),
        "total_txts": len(txt_files),
        "successful": len(results),
        "failed": len(errors),
        "documents": [
            {
                "agreement_id": r["agreement_id"],
                "filename": r["filename"],
                "txt_path": r["txt_path"],
                "page_count": r["page_count"],
                "char_count": r["char_count"],
                "metadata": r["metadata"],
            }
            for r in results
        ],
        "errors": errors,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest written: {manifest_path}")

    logger.info(
        f"Done: {len(results)}/{len(all_files)} files parsed, "
        f"{len(errors)} errors"
    )
    return results


def load_parsed_agreements(
    parsed_dir: str | Path | None = None,
) -> list[dict]:
    """
    Load previously parsed agreements from the output directory.

    Reads manifest.json and returns agreement dicts compatible with
    run_full_pipeline(agreements=...).
    """
    parsed_dir = Path(parsed_dir) if parsed_dir else PARSED_TEXT_DIR
    manifest_path = parsed_dir / "manifest.json"

    if not manifest_path.exists():
        logger.warning(f"No manifest found at {manifest_path}")
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    agreements = []
    for doc in manifest.get("documents", []):
        txt_path = Path(doc["txt_path"])
        if txt_path.exists():
            agreements.append({
                "agreement_id": doc["agreement_id"],
                "filename": doc["filename"],
                "raw_text": txt_path.read_text(encoding="utf-8"),
            })
        else:
            logger.warning(f"Missing text file: {txt_path}")

    logger.info(f"Loaded {len(agreements)} parsed agreements from {parsed_dir}")
    return agreements


# ── CLI entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m core.pdf_parser <input_dir> [--output <output_dir>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    parse_pdf_directory(input_path, output_path)
