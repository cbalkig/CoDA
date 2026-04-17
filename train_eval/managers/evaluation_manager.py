from data.configuration.configuration import Configuration
from data.dataset.data import Data
from data.stages.stages.eval.eval_model import EvaluateModelTypeStage
from managers.operation_manager import OperationManager
from model.model import Model
from util.device_detector import DeviceDetector


class EvaluationManager(OperationManager):
    def __init__(self, model: Model, data: Data, config: Configuration):
        super().__init__(model, data, config, EvaluateModelTypeStage(), None)

    def run(self):
        _ = self._execute_substages()
        self.finish()

        DeviceDetector().empty_cache()
