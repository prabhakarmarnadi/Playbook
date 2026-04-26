"""Run the clustering pipeline."""
import sys
import os
from pathlib import Path

# Add hackathon root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from core.cluster_pipeline import run_cluster_pipeline

run_cluster_pipeline(
    account_id="public_docs_semantic",
    pdf_dir="/home/azureuser/contract_data/5k_public_docs",
    max_docs=int(sys.argv[1]) if len(sys.argv) > 1 else 1000,
    device="cuda",
)
