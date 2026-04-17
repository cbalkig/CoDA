from enum import Enum


class ModelType(Enum):
    FEATURE_EXTRACTOR = "feature_extractor"
    CLASSIFIER = "classifier"

    def __str__(self):
        return self.name.lower()
