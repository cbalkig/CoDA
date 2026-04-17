# abstract_file_util.py
from __future__ import annotations

import errno
import fnmatch
import gc
import logging
import os
import pickle
import posixpath
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import PurePosixPath, Path
from typing import Any, Union, Optional, List, Tuple, Dict

import gcsfs
import joblib
import torch
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.cloud.storage import Bucket
from tqdm import tqdm

from data.file.gcs_parts import GCSParts
from data.file.path import StoragePath
from util.device_detector import DeviceDetector


class AbstractFileUtil:
    """
    File helper with explicit path semantics:

    - If a path is a `Path` instance => treat as LOCAL filesystem.
    - If a path is a `str`        => treat as a GCS URL starting with "gs://<bucket>/<key>".

    Notes:
    - For GCS, you can mix buckets per call (e.g., copy between different buckets).
    - `bucket_name` acts only as a DEFAULT; explicit `gs://other-bucket/...` on arguments wins.
    """

    def __init__(
            self,
            bucket_name: Optional[str] = None,
            credentials_path: Optional[Union[Path, str]] = None,
    ):
        self.default_bucket_name: Optional[Path] = bucket_name
        self.credentials_path: Optional[Path] = (
            str(credentials_path) if credentials_path is not None else None
        )

        # Only set env var if credentials are provided
        if self.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(self.credentials_path)

        self.storage_client: Optional[storage.Client] = None
        # IMPORTANT: don't keep a single Bucket bound to a bucket_name;
        # resolve per-path to support cross-bucket operations.
        self.fs: Optional[gcsfs.GCSFileSystem] = None  # Lazy init for fork safety

    # ---------- Init helpers ----------

    def _initialize_storage(self) -> None:
        if self.storage_client is None:
            self.storage_client = storage.Client()
        if self.fs is None:
            # gcsfs accepts: "google_default", path to keyfile, dict, or token=None
            token = self.credentials_path if self.credentials_path else "google_default"
            self.fs = gcsfs.GCSFileSystem(token=token)

    # ---------- Path helpers ----------
    def _bucket(self, bucket_name: Optional[str]) -> Bucket:
        self._initialize_storage()
        name = bucket_name or self.default_bucket_name
        if not name:
            raise ValueError(
                "No bucket specified. Provide a GCS URL with bucket "
                "or set a default bucket_name on the class."
            )
        return self.storage_client.bucket(name)

    # ---------- Small utils ----------

    @staticmethod
    def _safe_unlink(path: Path, retries: int = 5, delay: float = 0.2) -> None:
        for _ in range(retries):
            try:
                path.unlink()
                return
            except PermissionError as e:
                if e.errno not in (errno.EACCES, errno.EPERM):
                    raise
                gc.collect()
                time.sleep(delay)
                delay *= 2
        path.unlink()

    # ---------- Simple IO ----------

    def read_file(self, file_path: StoragePath) -> bytes:
        if file_path.local:
            with open(file_path.path, "rb") as f:
                return f.read()
        # GCS
        bucket = self._bucket(file_path.bucket)
        return bucket.blob(file_path.key).download_as_bytes()

    def write_file(self, file_path: StoragePath, data: Union[str, bytes], mode: str = "w") -> None:
        if file_path.local:
            if isinstance(data, bytes):
                open_mode = "wb"
                payload = data
            else:
                open_mode = mode
                payload = data
            Path(file_path.path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path.path, open_mode) as f:
                f.write(payload)
            return

        # GCS
        bucket = self._bucket(file_path.bucket)
        blob = bucket.blob(file_path.key)
        if isinstance(data, bytes):
            blob.upload_from_string(data)
        else:
            blob.upload_from_string(data.encode("utf-8"))

    # ---------- Listing / existence ----------

    def list_files(self, directory: StoragePath) -> List[str]:
        if directory.local:
            return [str(p) for p in Path(directory.path).iterdir()]

        client = self._bucket(directory.bucket).client
        blobs = client.list_blobs(directory.bucket, prefix=directory.key)
        return [f"gs://{directory.bucket}/{b.name}" for b in blobs]

    def exists(self, file_path: StoragePath) -> bool:
        if file_path.local:
            return Path(file_path.path).exists()

        bucket = self._bucket(file_path.bucket)
        blob = bucket.blob(file_path.key)
        if blob.exists():
            return True

        # folder-like check
        prefix = file_path.key.rstrip("/") + "/"
        client = bucket.client
        blobs = list(client.list_blobs(file_path.bucket, prefix=prefix, max_results=1))
        return len(blobs) > 0

    def file_exists(self, file_path: StoragePath) -> bool:
        # Back-compat name
        return self.exists(file_path)

    # ---------- Gather with pattern & extensions ----------

    def gather_files(self, directory: StoragePath, pattern: str, extensions: List[str]) -> List[
        StoragePath]:
        extensions = [e.lower() for e in extensions]
        if directory.local:
            root = Path(directory.path)
            return [
                StoragePath(Path(p))
                for p in root.glob(pattern)
                if p.is_file() and p.suffix.lower() in extensions
            ]

        # GCS
        client = self._bucket(directory.bucket).client

        # `pattern` will be matched against the path RELATIVE to the provided prefix root
        # Determine a fixed prefix to list under (roughly the "directory" portion before any wildcard)
        prefix_root = directory.key.rstrip("/")
        if "*" in prefix_root or "?" in prefix_root or "[" in prefix_root:
            # If directory itself contains wildcards (rare), fallback to the highest stable prefix
            # (up to first glob metachar). This keeps listing bounded.
            cut = min([i for i in [prefix_root.find("*"), prefix_root.find("?"), prefix_root.find("[")] if i != -1],
                      default=len(prefix_root))
            stable_prefix = prefix_root[:cut]
        else:
            stable_prefix = prefix_root + "/"

        out: List[StoragePath] = []
        for blob in client.list_blobs(directory.bucket, prefix=stable_prefix):
            rel = blob.name[len(stable_prefix):].lstrip("/")
            if rel and fnmatch.fnmatch(rel, pattern) and any(
                    rel.lower().endswith(ext) for ext in extensions
            ):
                out.append(StoragePath(f"gs://{directory.bucket}/{blob.name}"))
        return out

    # ---------- Mutations ----------

    def delete_file(self, file_path: StoragePath) -> None:
        if file_path.local:
            Path(file_path.path).unlink(missing_ok=True)
            return

        self._bucket(file_path.bucket).blob(file_path.key).delete()

    @staticmethod
    def create_directory(directory_path: StoragePath, clean: bool = False) -> None:
        if directory_path.local:
            p = directory_path.path
            if clean and p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
        else:
            return

    # ---------- Copy helpers ----------

    def _copy_gcs_to_gcs(self, src: GCSParts, dst: GCSParts) -> None:
        client = self._bucket(src.bucket).client
        src_bucket = self._bucket(src.bucket)
        dst_bucket = self._bucket(dst.bucket)

        # If src.key ends with '/', treat as prefix copy
        is_prefix = src.key.endswith("/")
        if is_prefix:
            for blob in client.list_blobs(src.bucket, prefix=src.key):
                rel = blob.name[len(src.key):]
                if not rel:
                    continue
                new_key = dst.key.rstrip("/") + "/" + rel
                src_blob = src_bucket.blob(blob.name)
                src_bucket.copy_blob(src_blob, dst_bucket, new_key)
        else:
            # single object copy
            src_blob = src_bucket.blob(src.key)
            dst_key = dst.key if dst.key else posixpath.basename(src.key)
            src_bucket.copy_blob(src_blob, dst_bucket, dst_key)

    def _download_gcs_folder(self, src: GCSParts, local_dst: Path, num_threads: int = 32) -> None:
        self._initialize_storage()
        assert self.fs is not None

        prefix = f"{src.bucket}/{src.key.rstrip('/')}"
        all_files = self.fs.find(prefix)

        def _dl(remote_file: str) -> None:
            relative_path = os.path.relpath(remote_file, prefix).lstrip("./")
            local_file = local_dst / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            with self.fs.open(remote_file, "rb") as src_f, open(local_file, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)

        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = {ex.submit(_dl, f): f for f in all_files}
            for _ in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Downloading Files from {src.bucket}/{src.key}"):
                pass

    @staticmethod
    def _build_dest_key(prefix: str, local_src: Path, file_path: Path) -> str:
        rel = file_path.relative_to(local_src).as_posix()
        return f"{prefix}/{rel}" if prefix else rel

    @staticmethod
    def _upload_one(
            bucket: Bucket,
            local_src: Path,
            file_path: Path,
            prefix: str,
            chunk_size: Optional[int] = None,
    ) -> Tuple[Path, Optional[Exception]]:
        try:
            dest_key = AbstractFileUtil()._build_dest_key(prefix, local_src, file_path)
            blob = bucket.blob(dest_key)
            if chunk_size:
                # For large files you can tune this (must be a multiple of 256 KB)
                blob.chunk_size = chunk_size
            blob.upload_from_filename(str(file_path))
            return file_path, None
        except Exception as e:
            return file_path, e

    @staticmethod
    def _gather_files(root: Path) -> List[Path]:
        # Faster than checking inside the loop
        return [p for p in root.rglob("*") if p.is_file()]

    # Parallel version
    def _upload_local_folder_to_gcs(
            self,
            local_src: Path,
            dst: GCSParts,
            *,
            max_workers: int = 8,
            chunk_size: Optional[int] = None,  # e.g. 8 * 1024 * 1024
            fail_fast: bool = False,  # raise on first failure if True
    ) -> None:
        bucket = self._bucket(dst.bucket)  # get once; client/bucket are thread-safe for I/O
        prefix = (dst.key or "").rstrip("/")

        files = self._gather_files(local_src)
        if not files:
            return

        errors: Dict[Path, Exception] = {}

        # Thread pool for I/O-bound uploads
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(self._upload_one, bucket, local_src, f, prefix, chunk_size)
                for f in files
            ]
            for fut in as_completed(futures):
                file_path, err = fut.result()
                if err is not None:
                    if fail_fast:
                        # cancel remaining tasks and raise
                        for other in futures:
                            other.cancel()
                        raise RuntimeError(f"GCS upload failed for {file_path}") from err
                    errors[file_path] = err

        if errors:
            # Aggregate exceptions but continue after uploading what we could
            msg_lines = ["Some files failed to upload:"]
            for p, e in list(errors.items())[:10]:
                msg_lines.append(f" - {p}: {e}")
            if len(errors) > 10:
                msg_lines.append(f" ... and {len(errors) - 10} more")
            raise RuntimeError("\n".join(msg_lines))

    # ---------- Public copy APIs ----------
    def copy_folder(self, source_path: StoragePath, destination_path: StoragePath, clean_first: bool = False) -> None:
        if source_path.local and destination_path.local:
            src: Path = source_path.path
            dst: Path = destination_path.path

            if src.resolve() == dst.resolve():
                return

            if src.is_dir():
                if clean_first and dst.exists():
                    shutil.rmtree(dst)

                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            return

        # Mixed/GCS cases
        if not source_path.local and not destination_path.local:
            self._copy_gcs_to_gcs(source_path.gcs_parts, destination_path.gcs_parts)
            return

        if not source_path.local and destination_path.local:
            # GCS -> local
            self._download_gcs_folder(source_path.gcs_parts, destination_path.path)
            return

        # local -> GCS
        if source_path.path.is_dir():
            self._upload_local_folder_to_gcs(source_path.path, destination_path.gcs_parts)
        else:
            key = destination_path.key or ""
            if not key or not key.endswith("/"):
                final_key = (key + "/" if key else "") + source_path.path.name
            else:
                final_key = key + source_path.path.name
            self._bucket(destination_path.bucket).blob(final_key).upload_from_filename(str(source_path.path))

    def copy_file(self, source_path: StoragePath, destination_path: StoragePath, delete_local: bool = False) -> None:
        if source_path.local and destination_path.local:
            if source_path.path.resolve() == destination_path.path.resolve():
                return
            destination_path.path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path.path, destination_path.path)
            if delete_local:
                self._safe_unlink(source_path.path)
            return

        if not source_path.local and not destination_path.local:
            self._copy_gcs_to_gcs(source_path.gcs_parts, destination_path.gcs_parts)
            if delete_local:
                # source is GCS; honor legacy behavior and delete remote source blob
                self._bucket(source_path.bucket).blob(source_path.key).delete()
            return

        if not source_path.local and destination_path.local:
            destination_path.path.parent.mkdir(parents=True, exist_ok=True)
            self._bucket(source_path.bucket).blob(source_path.key).download_to_filename(str(destination_path.path))
            if delete_local:
                # source is GCS; maintain legacy behavior
                self._bucket(source_path.bucket).blob(source_path.key).delete()
            return

        # local -> GCS
        self._bucket(destination_path.bucket).blob(destination_path.key).upload_from_filename(str(source_path.path))
        if delete_local:
            self._safe_unlink(source_path.path)

    def dump(self, obj: Any, file_path: StoragePath) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_path.suffix) as tmp_file:
            tmp_name = tmp_file.name
        try:
            if str(file_path).endswith((".joblib", ".pkl")):
                joblib.dump(obj, tmp_name)
            elif str(file_path).endswith(".pt"):
                torch.save(obj, tmp_name)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

            if file_path.local:
                file_path.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmp_name, str(file_path))
            else:
                self._bucket(file_path.bucket).blob(file_path.key).upload_from_filename(tmp_name)
        finally:
            if Path(tmp_name).exists():
                os.remove(tmp_name)

    def load(self, file_path: StoragePath, **kwargs) -> Any:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_path.suffix) as tmp_file:
            tmp_name = tmp_file.name

        try:
            if file_path.local:
                shutil.copy(str(file_path.path), tmp_name)
            else:
                self._bucket(file_path.bucket).blob(file_path.key).download_to_filename(tmp_name)

            if str(file_path.path).endswith((".joblib", ".pkl")):
                return joblib.load(tmp_name, **kwargs)
            elif str(file_path.path).endswith(".pt"):
                return torch.load(tmp_name, map_location=DeviceDetector().device, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        finally:
            if Path(tmp_name).exists():
                os.remove(tmp_name)

    def download_directory(self, remote_source: StoragePath, local_destination: Path, num_threads: int = 32) -> None:
        if remote_source.local:
            shutil.copytree(remote_source.path, local_destination)
            return
        self._download_gcs_folder(remote_source.gcs_parts, local_destination, num_threads=num_threads)

    def read_pickle_file(self, file_path: StoragePath) -> Any:
        if file_path.local:
            # Local read
            with open(file_path.path, "rb") as f:
                data = pickle.load(f)
        else:
            # Remote read (download first)
            content_bytes = self.download_remote(file_path)  # must return bytes
            data = pickle.load(BytesIO(content_bytes))

        return data

    def write_pickle_file(self, file_path: StoragePath, data: Any) -> None:
        if file_path.local:
            file_path.path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path.path, "wb") as f:
                pickle.dump(data, f)
        else:
            # Pickle to memory
            buffer = BytesIO()
            pickle.dump(data, buffer)
            buffer.seek(0)

            # Upload to remote
            self.upload_remote(file_path, buffer.getvalue())

    @staticmethod
    def download_remote(file_path: StoragePath) -> bytes:
        if file_path.local:
            raise ValueError("download_remote expects a remote StoragePath")

        try:
            if not file_path.key:
                raise ValueError(f"No object key provided in URL: {file_path.path!r}")

            client = storage.Client()
            bucket = client.bucket(file_path.bucket)
            blob = bucket.blob(file_path.key)

            # Download as bytes (raises NotFound if missing)
            return blob.download_as_bytes()
        except NotFound as e:
            raise FileNotFoundError(f"Remote object not found: {file_path.path}") from e

    @staticmethod
    def upload_remote(file_path: "StoragePath", content_bytes: bytes) -> None:
        if file_path.local:
            raise ValueError("upload_remote expects a remote StoragePath")

        if not file_path.key:
            raise ValueError(f"No object key provided in URL: {file_path.path!r}")

        client = storage.Client()
        bucket = client.bucket(file_path.bucket)
        blob = bucket.blob(file_path.key)

        blob.upload_from_string(content_bytes)

    @staticmethod
    def delete_directory(dir_path: StoragePath) -> None:
        if dir_path.local:
            if dir_path.path.exists():
                try:
                    shutil.rmtree(dir_path.path)
                except OSError as e:
                    logging.warning(f"Failed to remove directory {dir_path.path}: {e}")
        else:
            if not dir_path.key:
                raise ValueError(f"No object key (prefix) provided for remote directory: {dir_path.path!r}")

            client = storage.Client()
            bucket = client.bucket(dir_path.bucket)

            # List and delete all objects under the given prefix
            blobs = bucket.list_blobs(prefix=dir_path.key)
            for blob in blobs:
                blob.delete()

    def copy_files_with_extensions(self, source_folder: StoragePath, dest_folder: StoragePath, extensions: List[str],
                                   delete_old: bool = False) -> None:
        if not source_folder.local:
            raise Exception('Only supports copying files from local storage')

        if dest_folder.local:
            dest_folder.path.mkdir(parents=True, exist_ok=True)

        operate: List[Tuple[StoragePath, StoragePath]] = []
        for root, _, files in os.walk(source_folder.path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    source_path = StoragePath(os.path.join(root, file))
                    dest_path = dest_folder.join(file)

                    operate.append((source_path, dest_path))

        for item in operate:
            source_path, dest_path = item
            self.copy_file(source_path, dest_path)
            if delete_old:
                self.delete_file(source_path)

    def copy_folders_with_prefix(self, source_folder: StoragePath, dest_folder: StoragePath, prefixes: List[str]):
        prefixes = tuple(prefixes or ())

        if dest_folder.local:
            dest_folder.path.mkdir(parents=True, exist_ok=True)

        if source_folder.local:
            for entry in source_folder.path.iterdir():
                if entry.is_dir() and entry.name.startswith(prefixes):
                    src = StoragePath(entry)
                    dst = dest_folder.join(entry.name)
                    self.copy_folder(src, dst)
            return

        bucket = self._bucket(source_folder.bucket)
        client = bucket.client
        prefix = source_folder.key.rstrip("/") + "/"

        iterator = client.list_blobs(source_folder.bucket, prefix=prefix, delimiter="/")
        sub_prefixes = set()
        for page in iterator.pages:
            if getattr(page, "prefixes", None):
                sub_prefixes.update(page.prefixes)

        for sub in sorted(sub_prefixes):
            name = sub[len(prefix):].rstrip("/")
            if name and name.endswith(suffixes):
                src = StoragePath(f"gs://{source_folder.bucket}/{sub}")
                dst = dest_folder.join(name)
                self.copy_folder(src, dst)

    @staticmethod
    def unzip_file(zip_path: StoragePath, output_dir: Optional[StoragePath] = None) -> StoragePath:
        local_zip = None
        if not zip_path.local:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            local_zip = Path(tmp_file.name)
            tmp_file.close()

            # Download from GCS to local temp file
            client = storage.Client()
            bucket = client.bucket(zip_path.bucket)
            blob = bucket.blob(zip_path.key)
            blob.download_to_filename(str(local_zip))
            zip_file_path = local_zip
        else:
            zip_file_path = zip_path.path

        # Determine output folder
        if output_dir is None:
            output_dir = zip_file_path.parent / zip_file_path.stem
        output_dir.path.mkdir(parents=True, exist_ok=True)

        def _is_safe(member: str) -> bool:
            # prevent absolute paths and path traversal
            p = PurePosixPath(member)
            return (not member.startswith("/")) and (".." not in p.parts)

        with zipfile.ZipFile(zip_file_path, "r") as zf:
            # ignore directory entries and macOS metadata when detecting root
            file_members = [i for i in zf.infolist() if not i.is_dir()]
            safe_members = [m for m in file_members if _is_safe(m.filename)]
            if len(safe_members) != len(file_members):
                raise ValueError("Archive contains unsafe paths (absolute or '..').")

            def top_part(name: str) -> str | None:
                parts = PurePosixPath(name).parts
                if not parts:
                    return None
                # skip __MACOSX when detecting the single root folder
                return parts[0] if parts[0] != "__MACOSX" else (parts[1] if len(parts) > 1 else None)

            roots = {top_part(m.filename) for m in safe_members if top_part(m.filename)}
            omit_list = [".DS_Store", "__MACOSX"]
            for omit in omit_list:
                if omit in roots:
                    roots.remove(omit)

            # If exactly one real top-level directory, flatten it
            if len(roots) == 1:
                tmpdir = Path(tempfile.mkdtemp())
                try:
                    zf.extractall(tmpdir)
                    root = tmpdir / list(roots)[0]
                    # If the "single root" is actually a file, just move normally
                    if root.is_dir():
                        for item in root.iterdir():
                            dest = output_dir.path / item.name
                            # If exists, remove/make room (overwrite behavior)
                            if dest.exists():
                                if dest.is_dir():
                                    shutil.rmtree(dest)
                                else:
                                    dest.unlink()
                            shutil.move(str(item), str(dest))
                    else:
                        # Single file at root -> move it into output_dir
                        dest = output_dir.path / root.name
                        if dest.exists():
                            dest.unlink()
                        shutil.move(str(root), str(dest))
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            else:
                # Multiple top-level entries -> extract as-is into output_dir
                zf.extractall(output_dir.path)

        if local_zip and local_zip.exists():
            os.remove(local_zip)

        return output_dir

    def zip_folder(
            self,
            folder_path: StoragePath,
            zip_destination: StoragePath,
            *,
            include_root: bool = True,
            compression: int = zipfile.ZIP_DEFLATED,
    ) -> StoragePath:
        if not folder_path.local:
            raise ValueError("zip_folder currently supports zipping only from a local source folder.")

        root = folder_path.path
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Folder to zip does not exist or is not a directory: {root}")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            with zipfile.ZipFile(tmp_path, mode="w", compression=compression) as zf:
                base = root.parent if include_root else root
                for dirpath, dirnames, filenames in os.walk(root):
                    dirpath_p = Path(dirpath)

                    if include_root:
                        arc_dir = dirpath_p.relative_to(base).as_posix() + "/"
                        if arc_dir != "./":
                            zf.writestr(zipfile.ZipInfo(arc_dir), b"")

                    for fname in filenames:
                        fpath = dirpath_p / fname
                        arcname = fpath.relative_to(base).as_posix()
                        zf.write(fpath, arcname=arcname)

            if zip_destination.local:
                zip_destination.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_path), str(zip_destination.path))
            else:
                bucket = self._bucket(zip_destination.bucket)
                blob = bucket.blob(zip_destination.key)
                blob.upload_from_filename(str(tmp_path))
                self._safe_unlink(tmp_path)

            return zip_destination

        except Exception:
            if tmp_path.exists():
                try:
                    self._safe_unlink(tmp_path)
                except Exception:
                    pass
            raise
