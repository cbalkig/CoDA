from typing import Dict

from data.domain_losses import DomainLosses
from data.types.domain_type import DomainType


class Losses:
    def __init__(self, losses: Dict[DomainType, DomainLosses]) -> None:
        self.losses = losses

    def __str__(self) -> str:
        return (
            f"Losses:\n"
            f" - Losses:\n{self._format_nested(self.losses)}"
            f" - Weighted Total: {self.get_weighted_total():.10f}\n"
        )

    @staticmethod
    def _format_nested(losses) -> str:
        return "\n".join(f"   - {attr}: {getattr(losses, attr)}" for attr in vars(losses))

    def get_weighted_total(self) -> float:
        total = 0.0
        for domain_loss in self.losses.values():
            total += domain_loss.get_weighted_total()
        return total
