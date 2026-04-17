import os.path
from pathlib import Path
from typing import Optional

from data.file.gcs_parts import GCSParts


class StoragePath:
    def __init__(self, path: str | Path):
        self._local_path: Optional[Path] = None
        self._remote_path: Optional[str] = None
        self._remote: bool = False
        self._gcs_parts: Optional[GCSParts] = None

        if isinstance(path, Path):
            self._local_path = path
            self._remote = False
        elif isinstance(path, str):
            if path.startswith("gs://"):
                self._remote_path = path
                self._remote = True
                self._gcs_parts = self._parse_gcs(self._remote_path)
            else:
                self._local_path = Path(path)
                self._remote = False
        else:
            raise ValueError(f"Invalid storage path: {path!r} - Type{type(path)}")

    def __eq__(self, other):
        return isinstance(other,
                          StoragePath) and self._local_path == other._local_path and self._remote_path == other._remote_path and self._remote == other._remote

    def __str__(self) -> str:
        if self.local:
            return f'{Path(self.path).absolute()}'
        else:
            return f'{self.path}'

    @property
    def local(self) -> bool:
        return not self._remote

    @property
    def path(self) -> Path | str:
        if self.local:
            return self._local_path
        else:
            return self._remote_path

    @property
    def name(self) -> str:
        if self.local:
            return Path(self.path).name
        else:
            if not self.key:
                return self.bucket

            return Path(self.key).name

    @property
    def parent_name(self) -> str:
        if self.local:
            return Path(self.path).parent.name
        else:
            if not self.key or "/" not in self.key:
                return self.bucket

            key_path = Path(self.key.rstrip("/"))
            return key_path.parent.name

    @property
    def suffix(self) -> str:
        if self.local:
            return Path(self.path).suffix
        else:
            if not self.key:
                return ""
            return Path(self.key).suffix

    @property
    def bucket(self) -> str:
        return self._gcs_parts.bucket

    @property
    def key(self) -> str:
        return self._gcs_parts.key

    @property
    def gcs_parts(self) -> GCSParts:
        return self._gcs_parts

    @staticmethod
    def _parse_gcs(url: str) -> GCSParts:
        """
        Parse 'gs://bucket/key...' into (_GCSParts(bucket, key)).
        key can be '' (bucket root) or 'some/prefix' (no leading slash).
        """
        if not type(url) == str:
            raise ValueError(f"GCS path must be a string starting with 'gs://', got: {url!r}")

        GCSParts.assert_gcs_url(url)

        without = url[len("gs://"):]
        parts = without.split("/", 1)
        if not parts[0]:
            raise ValueError(f"Missing bucket in GCS URL: {url!r}")

        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return GCSParts(bucket=bucket, key=key)

    def join(self, *parts: str) -> "StoragePath":
        if self.local:
            return StoragePath(Path(self.path).joinpath(*parts))
        else:
            return StoragePath(os.path.join(self.path, *parts))
