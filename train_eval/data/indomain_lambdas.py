class InDomainLambdas:
    def __init__(self, classification: float) -> None:
        self.classification = classification

    def __str__(self) -> str:
        return (
                "In Domain Lambdas:\n" +
                "\n".join(f" - {attr}: {getattr(self, attr):.10f}" for attr in vars(self))
        )
