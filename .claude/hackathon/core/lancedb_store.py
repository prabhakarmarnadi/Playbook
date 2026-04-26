"""
LanceDB vector store — replaces FAISS for persistent, filterable vector search.

Legacy: AIDB gRPC ANN search (remote, single-metric).
Hackathon V1: FAISS IndexFlatIP (in-memory, no filtering, no persistence).
V2: LanceDB (on-disk, metadata filtering, persistent, IVF+PQ indexing).

Key advantages over FAISS:
  - Persistent — survives restarts without re-indexing
  - Filterable — filter by domain_id, agreement_id, etc. during search
  - Columnar — stores metadata alongside vectors (no separate lookup)
  - Scalable — IVF-PQ indexing for 100K+ vectors
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


class LanceVectorStore:
    """
    LanceDB-backed vector store for chunk embeddings.

    Tables:
      - chunks: chunk_id, agreement_id, domain_id, cluster_id, text, vector
      - documents: agreement_id, filename, summary, vector
    """

    def __init__(self, db_path: str | Path):
        import lancedb

        self.db_path = str(db_path)
        self.db = lancedb.connect(self.db_path)
        logger.info(f"LanceDB connected: {self.db_path}")

    def _get_or_create_table(
        self,
        name: str,
        schema: pa.Schema,
    ):
        """Get existing table or create with schema."""
        if name in self.db.table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    # ── Chunk vectors ──────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        agreement_ids: list[str],
        domain_ids: list[str] | None = None,
        cluster_ids: list[int] | None = None,
    ) -> int:
        """
        Insert or update chunk vectors.

        Args:
            chunk_ids: Unique IDs per chunk.
            vectors: (N, dim) float32 array.
            texts: Chunk text content.
            agreement_ids: Parent agreement IDs.
            domain_ids: Optional domain assignments.
            cluster_ids: Optional cluster assignments.

        Returns:
            Number of vectors upserted.
        """
        n = len(chunk_ids)
        _domain_ids = domain_ids or [""] * n
        _cluster_ids = cluster_ids or [-1] * n
        data = [
            {
                "chunk_id": chunk_ids[i],
                "agreement_id": agreement_ids[i],
                "domain_id": _domain_ids[i],
                "cluster_id": _cluster_ids[i],
                "text": texts[i],
                "vector": vectors[i].tolist(),
            }
            for i in range(n)
        ]

        table_name = "chunks"
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(data)
        else:
            self.db.create_table(table_name, data)

        logger.info(f"Upserted {n} chunk vectors")
        return n

    def search_chunks(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        ANN search over chunk vectors with optional SQL filter.

        Args:
            query_vector: (dim,) float32 query embedding.
            top_k: Number of nearest neighbors.
            where: Optional SQL WHERE clause (e.g., "domain_id = 'domain_1'").

        Returns:
            List of result dicts with chunk_id, text, score, metadata.
        """
        table = self.db.open_table("chunks")
        query = table.search(query_vector.tolist()).limit(top_k)
        if where:
            query = query.where(where)

        results = query.to_pandas()
        out = []
        for _, row in results.iterrows():
            out.append({
                "chunk_id": row["chunk_id"],
                "agreement_id": row["agreement_id"],
                "domain_id": row.get("domain_id", ""),
                "cluster_id": row.get("cluster_id", -1),
                "text": row["text"],
                "score": float(row.get("_distance", 0.0)),
            })
        return out

    # ── Document vectors ───────────────────────────────────────────────────

    def upsert_documents(
        self,
        agreement_ids: list[str],
        vectors: np.ndarray,
        filenames: list[str],
        summaries: list[str],
    ) -> int:
        """Insert or update document-level vectors."""
        data = [
            {
                "agreement_id": agreement_ids[i],
                "filename": filenames[i],
                "summary": summaries[i],
                "vector": vectors[i].tolist(),
            }
            for i in range(len(agreement_ids))
        ]

        table_name = "documents"
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(data)
        else:
            self.db.create_table(table_name, data)

        logger.info(f"Upserted {len(agreement_ids)} document vectors")
        return len(agreement_ids)

    def search_documents(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """ANN search over document-level vectors."""
        table = self.db.open_table("documents")
        results = table.search(query_vector.tolist()).limit(top_k).to_pandas()
        out = []
        for _, row in results.iterrows():
            out.append({
                "agreement_id": row["agreement_id"],
                "filename": row.get("filename", ""),
                "summary": row.get("summary", ""),
                "score": float(row.get("_distance", 0.0)),
            })
        return out

    # ── Utilities ──────────────────────────────────────────────────────────

    def count(self, table_name: str = "chunks") -> int:
        """Count rows in a table."""
        if table_name not in self.db.table_names():
            return 0
        return self.db.open_table(table_name).count_rows()

    def drop_table(self, table_name: str):
        """Drop a table."""
        if table_name in self.db.table_names():
            self.db.drop_table(table_name)
            logger.info(f"Dropped table: {table_name}")

    def list_tables(self) -> list[str]:
        """List all tables."""
        return self.db.table_names()

    def update_cluster_assignments(
        self,
        chunk_ids: list[str],
        domain_ids: list[str],
        cluster_ids: list[int],
    ):
        """
        Update domain_id and cluster_id for existing chunks.

        Used after clustering to annotate vectors with their assignments.
        """
        if "chunks" not in self.db.table_names():
            logger.warning("No chunks table — cannot update assignments")
            return

        table = self.db.open_table("chunks")
        for cid, did, clid in zip(chunk_ids, domain_ids, cluster_ids):
            table.update(
                where=f"chunk_id = '{cid}'",
                values={"domain_id": did, "cluster_id": clid},
            )
        logger.info(f"Updated cluster assignments for {len(chunk_ids)} chunks")

    # ── Backup / Restore ──────────────────────────────────────────────────

    def backup(self, backup_path: str | Path | None = None) -> Path:
        """
        Create a snapshot of the LanceDB directory.

        LanceDB stores data as immutable Arrow/Lance fragment files,
        so a directory copy is a consistent snapshot.

        Args:
            backup_path: Target directory. Defaults to
                         <lance_dir>_backup_<timestamp>/.
        Returns:
            Path to the backup directory.
        """
        import shutil
        from datetime import datetime

        if backup_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(f"{self.db_path}_backup_{ts}")
        backup_path = Path(backup_path)
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.copytree(self.db_path, backup_path)
        logger.info(f"LanceDB backup: {self.db_path} → {backup_path}")
        return backup_path

    @classmethod
    def restore(cls, backup_path: str | Path, db_path: str | Path) -> "LanceVectorStore":
        """
        Restore LanceDB from a backup directory.

        Args:
            backup_path: Directory produced by backup().
            db_path: Target path for the restored LanceDB.
        Returns:
            A new LanceVectorStore connected to the restored data.
        """
        import shutil

        backup_path = Path(backup_path)
        db_path = Path(db_path)
        if db_path.exists():
            shutil.rmtree(db_path)
        shutil.copytree(backup_path, db_path)
        logger.info(f"LanceDB restored: {backup_path} → {db_path}")
        return cls(db_path)
