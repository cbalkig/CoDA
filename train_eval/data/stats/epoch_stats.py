from abc import ABC


class EpochStats(ABC):
    @staticmethod
    def _format_nested(obj) -> str:
        return "\n\t".join(str(obj).splitlines())
