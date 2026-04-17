from data.lambdas import Lambdas
from data.loss.loss import Loss


class DomainLosses:
    def __init__(self, lambdas: Lambdas) -> None:
        self.lambdas = lambdas.in_domain
        self.classification = Loss()

    def __str__(self) -> str:
        text = "Domain Losses:\n"
        for attr in ["classification"]:
            loss = getattr(self, attr)
            avg = loss.get_average()
            if avg is not None:
                text += f" - {attr}: {avg:.10f}\n"
        text += f" - Total: {self.get_weighted_total():.10f}\n"
        return text

    @staticmethod
    def _get_weighted_loss(loss: Loss, weight: float) -> float:
        avg = loss.get_average()
        return (avg * weight) if avg is not None else 0.0

    def get_weighted_total(self) -> float:
        return (
            self._get_weighted_loss(self.classification, self.lambdas.classification)
        )
