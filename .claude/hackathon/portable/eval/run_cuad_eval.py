#!/usr/bin/env python3
"""
End-to-end CUAD evaluation script.

Downloads the CUAD dataset, ingests PDFs through the pipeline on CPU,
then runs the full eval suite.

Usage:
    # Default: download CUAD, ingest, evaluate (CPU)
    python scripts/run_cuad_eval.py

    # Skip download if already done
    python scripts/run_cuad_eval.py --skip-download

    # Skip ingest if cuad.duckdb already exists
    python scripts/run_cuad_eval.py --skip-ingest

    # Limit to N contracts (faster iteration)
    python scripts/run_cuad_eval.py --max-docs 50

    # Force GPU if available
    python scripts/run_cuad_eval.py --device cuda

    # Run with LLM judge (requires OPENAI_API_KEY)
    python scripts/run_cuad_eval.py --judge --judge-sample 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Ensure hackathon/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cuad_eval")

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CUAD_DIR = DATA_DIR / "cuad"
CUAD_ANNOTATIONS = CUAD_DIR / "CUADv1.json"
CUAD_DB = DATA_DIR / "cuad_eval.duckdb"
CUAD_LANCE = DATA_DIR / "cuad_eval.lance"
EVAL_OUTPUT = Path("eval") / "report_cuad.md"
EVAL_BASELINE = Path("eval") / "baseline_cuad.json"

CUAD_DATA_ZIP_URL = "https://raw.githubusercontent.com/TheAtticusProject/cuad/master/data.zip"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Download CUAD Annotations
# ═══════════════════════════════════════════════════════════════════════════════

def download_cuad(force: bool = False) -> Path:
    """Download CUAD annotations from the official GitHub repo.
    
    Downloads data.zip which contains CUADv1.json (SQuAD-format annotations
    with full contract text in the 'context' fields). No PDFs needed.
    """
    if CUAD_ANNOTATIONS.exists() and not force:
        size_mb = CUAD_ANNOTATIONS.stat().st_size / 1e6
        if size_mb > 1:
            logger.info(f"CUAD annotations already downloaded: {CUAD_ANNOTATIONS} ({size_mb:.1f} MB)")
            return CUAD_ANNOTATIONS

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CUAD_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = CUAD_DIR / "data.zip"

    logger.info("Downloading CUAD data from GitHub...")
    logger.info(f"  URL: {CUAD_DATA_ZIP_URL}")

    ret = subprocess.run(
        ["curl", "-sL", "-o", str(zip_path), CUAD_DATA_ZIP_URL],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"Download failed: {ret.stderr}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CUAD_DIR)

    zip_path.unlink(missing_ok=True)

    size_mb = CUAD_ANNOTATIONS.stat().st_size / 1e6
    logger.info(f"  Extracted CUADv1.json ({size_mb:.1f} MB) → {CUAD_ANNOTATIONS}")
    return CUAD_ANNOTATIONS


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Build Gold Standard from CUAD Annotations
# ═══════════════════════════════════════════════════════════════════════════════

def build_cuad_gold_set(max_docs: int | None = None) -> Path:
    """
    Convert CUAD SQuAD-format annotations into our gold_standard.json format.
    
    CUAD has 41 clause categories with span annotations — map them to our
    cluster_ground_truth and intent_ground_truth structures.
    """
    gold_path = Path("eval") / "gold_standard_cuad.json"
    
    if not CUAD_ANNOTATIONS.exists():
        logger.warning("No CUAD annotations found — running unsupervised eval only")
        return None
    
    logger.info("Building gold standard from CUAD annotations...")
    
    with open(CUAD_ANNOTATIONS) as f:
        cuad = json.load(f)
    
    # CUAD is in SQuAD format: {"data": [{"title": ..., "paragraphs": [...]}]}
    documents = cuad.get("data", [])
    if max_docs:
        documents = documents[:max_docs]
    
    # CUAD's 41 clause categories become our clustering ground truth
    cluster_ground_truth = {}
    intent_ground_truth = {}
    gold_documents = []
    
    for doc in documents:
        title = doc.get("title", "")
        doc_entry = {
            "agreement_id": title,
            "filename": f"{title}.pdf",
            "clauses": [],
        }
        
        for para in doc.get("paragraphs", []):
            context = para.get("context", "")
            
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                is_impossible = qa.get("is_impossible", True)
                
                if is_impossible or not answers:
                    continue
                
                # The question encodes the clause type (e.g., "Highlight the parts...")
                # Extract category from the question ID
                qa_id = qa.get("id", "")
                
                # Use the answer text as the clause span
                for ans in answers:
                    answer_text = ans.get("text", "").strip()
                    if not answer_text or len(answer_text) < 10:
                        continue
                    
                    clause_id = f"{title}__{qa_id}__{ans.get('answer_start', 0)}"
                    
                    # Map question to clause category
                    category = _extract_cuad_category(question)
                    cluster_ground_truth[clause_id] = category
                    
                    # Build intent ground truth
                    intent_ground_truth[clause_id] = [{
                        "label": category,
                        "summary": answer_text[:200],
                        "party_from": "",
                        "party_to": "",
                    }]
                    
                    doc_entry["clauses"].append({
                        "clause_id": clause_id,
                        "text": answer_text,
                        "gold_category": category,
                        "gold_intents": [{
                            "label": category,
                            "summary": answer_text[:200],
                        }],
                    })
        
        if doc_entry["clauses"]:
            gold_documents.append(doc_entry)
    
    gold_set = {
        "_comment": "Auto-generated from CUAD v1 annotations",
        "_source": "TheAtticusProject/cuad CUADv1.json",
        "_n_documents": len(gold_documents),
        "_n_clauses": len(cluster_ground_truth),
        "_n_categories": len(set(cluster_ground_truth.values())),
        "cluster_ground_truth": cluster_ground_truth,
        "field_ground_truth": {},
        "intent_ground_truth": intent_ground_truth,
        "documents": gold_documents,
    }
    
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_path, "w") as f:
        json.dump(gold_set, f, indent=2)
    
    categories = set(cluster_ground_truth.values())
    logger.info(
        f"Gold set built: {len(gold_documents)} docs, "
        f"{len(cluster_ground_truth)} annotations, "
        f"{len(categories)} clause categories"
    )
    return gold_path


def _extract_cuad_category(question: str) -> str:
    """
    Map CUAD question text to a clause category label.
    
    CUAD questions follow patterns like:
    "Highlight the parts (if any) of this contract related to 'Change of Control'"
    """
    # Known CUAD categories (41 total)
    CUAD_CATEGORIES = [
        "Document Name", "Parties", "Agreement Date", "Effective Date",
        "Expiration Date", "Renewal Term", "Notice Period To Terminate Renewal",
        "Governing Law", "Most Favored Nation", "Non-Compete",
        "Exclusivity", "No-Solicit Of Customers", "No-Solicit Of Employees",
        "Non-Disparagement", "Termination For Convenience",
        "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment",
        "Revenue/Profit Sharing", "Price Restrictions",
        "Minimum Commitment", "Volume Restriction", "Ip Ownership Assignment",
        "Joint Ip Ownership", "License Grant", "Non-Transferable License",
        "Affiliate License-Licensor", "Affiliate License-Licensee",
        "Unlimited/All-You-Can-Eat-License", "Irrevocable Or Perpetual License",
        "Source Code Escrow", "Post-Termination Services",
        "Competitive Restriction Exception", "Covenant Not To Sue",
        "Third Party Beneficiary", "Liquidated Damages",
        "Warranty Duration", "Insurance", "Cap On Liability",
        "Uncapped Liability", "Audit Rights",
    ]
    
    q_lower = question.lower()
    for cat in CUAD_CATEGORIES:
        if cat.lower().replace("-", " ").replace("/", " ") in q_lower.replace("'", "").replace("-", " ").replace("/", " "):
            return cat
    
    # Fallback: extract quoted text from question
    import re
    match = re.search(r"['\"]([^'\"]+)['\"]", question)
    if match:
        return match.group(1).title()
    
    return "Unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Ingest CUAD contracts through EVoC pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _write_cuad_texts(max_docs: int | None = None) -> Path:
    """Extract contract texts from CUAD JSON and write as .txt files for EVoC.
    
    CUAD stores full contract text in the 'context' fields of its SQuAD-format
    JSON. We concatenate contexts per document and write one .txt per contract.
    """
    text_dir = DATA_DIR / "cuad_texts"
    
    # Check if already materialized
    if text_dir.exists():
        existing = list(text_dir.glob("*.txt"))
        if existing:
            if max_docs and len(existing) > max_docs:
                existing = existing[:max_docs]
            logger.info(f"Using {len(existing)} cached .txt files in {text_dir}")
            return text_dir
    
    text_dir.mkdir(parents=True, exist_ok=True)
    
    with open(CUAD_ANNOTATIONS) as f:
        cuad = json.load(f)
    
    documents = cuad.get("data", [])
    if max_docs:
        documents = documents[:max_docs]
    
    n_written = 0
    for doc in documents:
        title = doc.get("title", f"doc_{n_written}")
        # Sanitize filename
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
        
        # Concatenate all paragraph contexts (these are the contract text segments)
        paragraphs = doc.get("paragraphs", [])
        full_text = "\n\n".join(p.get("context", "") for p in paragraphs)
        
        if len(full_text.strip()) < 100:
            continue
        
        out_path = text_dir / f"{safe_title}.txt"
        with open(out_path, "w") as f:
            f.write(full_text)
        n_written += 1
    
    logger.info(f"Wrote {n_written} contract .txt files to {text_dir}")
    return text_dir


def ingest_cuad(device: str, max_docs: int | None = None) -> Path:
    """Run the EVoC pipeline on CUAD contracts."""
    if CUAD_DB.exists():
        import duckdb
        conn = duckdb.connect(str(CUAD_DB), read_only=True)
        try:
            n = conn.execute("SELECT COUNT(*) FROM agreements").fetchone()[0]
            if n > 0:
                logger.info(f"Existing DB has {n} agreements — skipping ingest (delete {CUAD_DB} to re-run)")
                return CUAD_DB
        except Exception:
            pass
        finally:
            conn.close()
    
    if not CUAD_ANNOTATIONS.exists():
        raise FileNotFoundError(f"CUAD annotations not found at {CUAD_ANNOTATIONS}. Run download first.")
    
    # Write contract texts from CUAD JSON → .txt files for the EVoC pipeline
    text_dir = _write_cuad_texts(max_docs=max_docs)
    n_files = len(list(text_dir.glob("*.txt")))
    
    logger.info(f"Running EVoC pipeline on {n_files} contracts (device={device})...")
    t0 = time.time()
    
    from core.evoc_pipeline import run_evoc_pipeline
    
    result = run_evoc_pipeline(
        pdf_dir=str(text_dir),
        account_id="cuad_eval",
        mode="clause",
        db_path=str(CUAD_DB),
        lance_path=str(CUAD_LANCE),
        device=device,
        max_docs=max_docs,
        use_semantic_chunking=True,
        use_optuna=True,
        optuna_trials=15,  # fewer trials for eval speed
    )
    
    # Close the store connection so eval can open the DB
    if "store" in result and hasattr(result["store"], "close"):
        result["store"].close()
    elif "store" in result and hasattr(result["store"], "conn"):
        try:
            result["store"].conn.close()
        except Exception:
            pass
    
    elapsed = time.time() - t0
    logger.info(
        f"EVoC ingest complete in {elapsed:.0f}s: "
        f"{result.get('n_documents', 0)} docs, {result.get('n_chunks', 0)} chunks, "
        f"{result.get('n_clause_types', 0)} clause types"
    )
    return CUAD_DB


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Run evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(
    db_path: Path,
    gold_path: Path | None = None,
    run_judge: bool = False,
    judge_sample: int = 50,
    save_baseline: bool = False,
) -> dict:
    """Run the eval suite on the ingested CUAD database."""
    from eval.runner import run_evaluation
    
    logger.info("=" * 60)
    logger.info("  CUAD EVALUATION")
    logger.info("=" * 60)
    
    kwargs = {
        "db_path": str(db_path),
        "output_path": str(EVAL_OUTPUT),
        "embedding_dim": 768,
    }
    
    if gold_path and gold_path.exists():
        kwargs["gold_path"] = str(gold_path)
        logger.info(f"  Gold set: {gold_path}")
    
    baseline = EVAL_BASELINE if EVAL_BASELINE.exists() else None
    if baseline:
        kwargs["baseline_path"] = str(baseline)
        logger.info(f"  Baseline: {baseline}")
    
    if save_baseline:
        kwargs["save_baseline_path"] = str(EVAL_BASELINE)
    
    if run_judge:
        kwargs["run_judge"] = True
        kwargs["judge_sample"] = judge_sample
        logger.info(f"  LLM Judge: ON (sample={judge_sample})")
    
    results = run_evaluation(**kwargs)
    
    # Print key metrics summary
    print("\n" + "=" * 60)
    print("  CUAD EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Health Score:       {results.get('health.score', 'N/A')}")
    print(f"  Health Grade:       {results.get('health.grade', 'N/A')}")
    print(f"  Silhouette:         {results.get('clustering.silhouette', 'N/A')}")
    print(f"  Outlier %:          {results.get('clustering.outlier_pct', 'N/A')}")
    print(f"  Coverage cascade:   {results.get('system.coverage_cascade', 'N/A')}")
    print(f"  Coherence (mean):   {results.get('clustering.coherence_mean', 'N/A')}")
    
    if gold_path:
        print(f"  Field F1:           {results.get('field.f1', 'N/A')}")
        print(f"  Intent sim (mean):  {results.get('intent_gold.similarity_mean', 'N/A')}")
    
    print(f"\n  Report: {EVAL_OUTPUT}")
    print("=" * 60)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end CUAD dataset evaluation for Clustering V2 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Limit number of CUAD contracts to ingest (for faster iteration)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip CUAD download (assume already present in data/cuad/)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip PDF ingest (assume data/cuad_eval.duckdb exists)",
    )
    parser.add_argument(
        "--skip-gold", action="store_true",
        help="Skip gold set generation (unsupervised metrics only)",
    )
    parser.add_argument(
        "--judge", action="store_true",
        help="Run LLM-as-a-Judge (requires OPENAI_API_KEY, costs tokens)",
    )
    parser.add_argument(
        "--judge-sample", type=int, default=50,
        help="Number of clauses to sample for judge (default: 50)",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save results as the new CUAD baseline",
    )
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {device}")
    else:
        device = args.device
    
    t0 = time.time()
    
    # Step 1: Download
    if not args.skip_download:
        download_cuad()
    else:
        logger.info("Skipping download (--skip-download)")
    
    # Step 2: Build gold set from CUAD annotations
    gold_path = None
    if not args.skip_gold:
        gold_path = build_cuad_gold_set(max_docs=args.max_docs)
    
    # Step 3: Ingest
    if not args.skip_ingest:
        db_path = ingest_cuad(device=device, max_docs=args.max_docs)
    else:
        db_path = CUAD_DB
        if not db_path.exists():
            logger.error(f"--skip-ingest but {db_path} does not exist!")
            sys.exit(1)
        logger.info(f"Skipping ingest — using {db_path}")
    
    # Step 4: Evaluate
    results = run_eval(
        db_path=db_path,
        gold_path=gold_path,
        run_judge=args.judge,
        judge_sample=args.judge_sample,
        save_baseline=args.save_baseline,
    )
    
    total = time.time() - t0
    logger.info(f"Total CUAD eval time: {total:.0f}s ({total/60:.1f} min)")
    
    return results


if __name__ == "__main__":
    main()
