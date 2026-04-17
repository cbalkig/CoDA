class EpochStageStatus:
    def __init__(
            self,
            mean_change: float = 0,
            variance: float = 0,
            delta: float = 0,
            patience: int = 0,
            stabilisation_counter: int = 0,
    ) -> None:
        self.mean_change = mean_change
        self.variance = variance
        self.delta = delta
        self.patience = patience
        self.stabilisation_counter = stabilisation_counter

    def __str__(self) -> str:
        text = "Train Epoch Status:\n"
        for attr in vars(self):
            value = getattr(self, attr)
            if value is not None:
                text += f" - {attr.replace('_', ' ').title()}: {value:.10f}\n" if isinstance(value,
                                                                                             float) else f" - {attr.replace('_', ' ').title()}: {value}\n"
        return text
