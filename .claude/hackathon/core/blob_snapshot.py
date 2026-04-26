"""
Azure Blob Storage snapshot manager for DuckDB + LanceDB.

Uploads local backup directories as compressed tar archives to Azure Blob,
and restores them back to local disk.

Usage:
    from core.blob_snapshot import BlobSnapshotManager

    mgr = BlobSnapshotManager()   # reads connection string from env/config

    # After a pipeline run:
    mgr.snapshot(store, lance_store, run_id="gpu_run_abc123")

    # List available snapshots:
    snapshots = mgr.list_snapshots()

    # Restore from blob:
    store, lance = mgr.restore(snapshot_name="20260401_143022")
"""
import io
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContainerClient

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER = "clustering-snapshots"


class BlobSnapshotManager:
    """Upload / download DuckDB + LanceDB snapshots to Azure Blob Storage."""

    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str | None = None,
        blob_prefix: str = "v2",
    ):
        """
        Args:
            connection_string: Azure Storage connection string.
                Falls back to AZURE_STORAGE_CONNECTION_STRING env var.
            container_name: Blob container for snapshots.
            blob_prefix: Optional prefix (folder) inside the container.
        """
        conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            raise ValueError(
                "Azure Storage connection string required. "
                "Set AZURE_STORAGE_CONNECTION_STRING or pass connection_string=."
            )
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container_name = container_name or os.getenv(
            "SNAPSHOT_CONTAINER", _DEFAULT_CONTAINER
        )
        self.blob_prefix = blob_prefix.strip("/")
        self._ensure_container()

    def _ensure_container(self):
        """Create the container if it doesn't exist."""
        container: ContainerClient = self.client.get_container_client(self.container_name)
        if not container.exists():
            container.create_container()
            logger.info(f"Created blob container: {self.container_name}")

    def _blob_name(self, snapshot_name: str, kind: str) -> str:
        """Build blob path: <prefix>/<snapshot_name>/duckdb.tar.gz or lance.tar.gz."""
        parts = [self.blob_prefix, snapshot_name, f"{kind}.tar.gz"]
        return "/".join(p for p in parts if p)

    # ── Upload ─────────────────────────────────────────────────────────────

    @staticmethod
    def _tar_directory(src_dir: Path) -> bytes:
        """Compress a directory into an in-memory tar.gz archive."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(src_dir), arcname=".")
        return buf.getvalue()

    def snapshot(
        self,
        store,
        lance_store,
        snapshot_name: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """
        Create a local backup of both databases and upload to Azure Blob.

        Args:
            store: ClusteringStore instance.
            lance_store: LanceVectorStore instance.
            snapshot_name: Name for this snapshot. Defaults to timestamp.
            run_id: Optional pipeline run ID to include in name.

        Returns:
            The snapshot name (can be passed to restore()).
        """
        if snapshot_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"{run_id}_{ts}" if run_id else ts

        container = self.client.get_container_client(self.container_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # DuckDB → Parquet export → tar.gz → blob
            duck_backup = store.backup(tmpdir / "duckdb_export")
            duck_blob = self._blob_name(snapshot_name, "duckdb")
            duck_bytes = self._tar_directory(duck_backup)
            container.upload_blob(
                duck_blob, duck_bytes, overwrite=True,
            )
            logger.info(
                f"Uploaded DuckDB snapshot: {duck_blob} "
                f"({len(duck_bytes) / 1024:.0f} KB)"
            )

            # LanceDB → directory copy → tar.gz → blob
            lance_backup = lance_store.backup(tmpdir / "lance_export")
            lance_blob = self._blob_name(snapshot_name, "lance")
            lance_bytes = self._tar_directory(lance_backup)
            container.upload_blob(
                lance_blob, lance_bytes, overwrite=True,
            )
            logger.info(
                f"Uploaded LanceDB snapshot: {lance_blob} "
                f"({len(lance_bytes) / 1024:.0f} KB)"
            )

        logger.info(f"Snapshot '{snapshot_name}' uploaded to {self.container_name}")
        return snapshot_name

    # ── Download / Restore ─────────────────────────────────────────────────

    @staticmethod
    def _untar_to(archive_bytes: bytes, dest_dir: Path):
        """Extract a tar.gz archive to a destination directory."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO(archive_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            # Security: reject paths that escape the dest directory
            for member in tar.getmembers():
                member_path = (dest_dir / member.name).resolve()
                if not str(member_path).startswith(str(dest_dir.resolve())):
                    raise ValueError(
                        f"Tar archive contains path traversal: {member.name}"
                    )
            tar.extractall(dest_dir, filter="data")

    def restore(
        self,
        snapshot_name: str,
        db_path: str | Path | None = None,
        lance_path: str | Path | None = None,
    ) -> tuple:
        """
        Download a snapshot from Azure Blob and restore both databases.

        Args:
            snapshot_name: Name returned by snapshot().
            db_path: Path for the restored DuckDB file.
                Defaults to config.DB_PATH.
            lance_path: Path for the restored LanceDB directory.
                Defaults to config.LANCEDB_PATH.

        Returns:
            (ClusteringStore, LanceVectorStore) connected to restored data.
        """
        from config import DB_PATH, LANCEDB_PATH
        from core.store import ClusteringStore
        from core.lancedb_store import LanceVectorStore

        db_path = Path(db_path) if db_path else Path(DB_PATH)
        lance_path = Path(lance_path) if lance_path else Path(LANCEDB_PATH)
        container = self.client.get_container_client(self.container_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Download and extract DuckDB
            duck_blob = self._blob_name(snapshot_name, "duckdb")
            duck_data = container.download_blob(duck_blob).readall()
            duck_export = tmpdir / "duckdb_export"
            self._untar_to(duck_data, duck_export)
            store = ClusteringStore.restore(duck_export, db_path)
            logger.info(f"Restored DuckDB from {duck_blob}")

            # Download and extract LanceDB
            lance_blob = self._blob_name(snapshot_name, "lance")
            lance_data = container.download_blob(lance_blob).readall()
            lance_export = tmpdir / "lance_export"
            self._untar_to(lance_data, lance_export)
            lance_store = LanceVectorStore.restore(lance_export, lance_path)
            logger.info(f"Restored LanceDB from {lance_blob}")

        logger.info(f"Snapshot '{snapshot_name}' fully restored")
        return store, lance_store

    # ── List / Delete ──────────────────────────────────────────────────────

    def list_snapshots(self) -> list[dict]:
        """
        List all available snapshots in the container.

        Returns:
            List of dicts with name, timestamp, duckdb_size, lance_size.
        """
        container = self.client.get_container_client(self.container_name)
        prefix = f"{self.blob_prefix}/" if self.blob_prefix else ""
        blobs = list(container.list_blobs(name_starts_with=prefix))

        # Group by snapshot name
        snapshots: dict[str, dict] = {}
        for blob in blobs:
            parts = blob.name.split("/")
            # prefix/snapshot_name/kind.tar.gz
            if len(parts) < 2:
                continue
            snap_name = parts[-2]
            kind = parts[-1].replace(".tar.gz", "")
            if snap_name not in snapshots:
                snapshots[snap_name] = {
                    "name": snap_name,
                    "timestamp": blob.last_modified,
                }
            snapshots[snap_name][f"{kind}_size_mb"] = round(
                (blob.size or 0) / (1024 * 1024), 2
            )

        result = sorted(snapshots.values(), key=lambda s: s.get("timestamp", ""), reverse=True)
        return result

    def delete_snapshot(self, snapshot_name: str):
        """Delete a snapshot (both duckdb and lance archives) from blob."""
        container = self.client.get_container_client(self.container_name)
        for kind in ("duckdb", "lance"):
            blob_name = self._blob_name(snapshot_name, kind)
            try:
                container.delete_blob(blob_name)
                logger.info(f"Deleted {blob_name}")
            except Exception:
                logger.warning(f"Blob not found: {blob_name}")
