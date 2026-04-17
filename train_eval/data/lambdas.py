from data.indomain_lambdas import InDomainLambdas


class Lambdas:
    def __init__(self, in_domain: InDomainLambdas, source: float, target: float) -> None:
        self.in_domain = in_domain
        self.source = source
        self.target = target

    def __str__(self) -> str:
        text = "Lambdas:\n"
        text += f" - source: {self.source:.10f}\n"
        text += f" - target: {self.target:.10f}\n"
        text += f" - in_domain:\n{self._format_nested(self.in_domain)}"
        return text

    def _format_nested(self, obj) -> str:
        return "\n".join(f"   - {attr}: {getattr(obj, attr):.10f}" for attr in vars(obj)) + "\n"
