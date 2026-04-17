from dataclasses import dataclass


@dataclass(frozen=True)
class GCSParts:
    bucket: str
    key: str  # may be "" (bucket root) or end with "/" to mean a prefix

    @staticmethod
    def assert_gcs_url(url: str) -> None:
        if not isinstance(url, str) or not url.startswith("gs://"):
            raise ValueError(
                f"GCS path must be a string starting with 'gs://', got: {url!r}"
            )
