import logging
from typing import Dict, KeysView, Any

from torch import nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, CosineAnnealingLR

from configs.base.configs import Configs
from configs.classifier import ClassifierConfig
from configs.feature_extractor import FeatureExtractorConfig
from data.file.path import StoragePath
from data.types.model_type import ModelType
from model.classifier.module import MLPClassifier
from model.feature_extractor.module import FeatureExtractor
from model.managers.dropout_manager import DropoutType
from util.device_detector import DeviceDetector
from util.file_util import FileUtil


class SubModel:
    def __init__(self, num_classes: int, dropouts: Dict[ModelType, Dict[DropoutType, float]],
                 learning_rates: Dict[ModelType, float]):
        super().__init__()

        self._schedulers = {}
        self._optimizers = {}
        self.num_classes = num_classes
        self.dropouts = dropouts
        self.learning_rates = learning_rates

        fe = FeatureExtractor(drop_rate=dropouts[ModelType.FEATURE_EXTRACTOR][DropoutType.DROPOUT],
                              drop_path_rate=dropouts[ModelType.FEATURE_EXTRACTOR][DropoutType.DROPOUT_PATH])
        cls = MLPClassifier(fe.output_dim, num_classes,
                            dropout=dropouts[ModelType.CLASSIFIER][DropoutType.DROPOUT])

        fe.freeze_all()
        if Configs().feature_extractor.unfrozen_all_layers:
            fe.unfreeze_last_n_layers(fe.number_of_layers)

        self._models: Dict[ModelType, nn.Module] = {ModelType.FEATURE_EXTRACTOR: DeviceDetector().to(fe),
                                                    ModelType.CLASSIFIER: DeviceDetector().to(cls), }

        fe_configs: FeatureExtractorConfig = Configs().feature_extractor
        cls_configs: ClassifierConfig = Configs().classifier

        if fe_configs.optimizer_type == AdamW:
            fe_optimizer = AdamW(
                fe.build_layer_wise_lr_decay_param_groups(base_lr=learning_rates[ModelType.FEATURE_EXTRACTOR],
                                                          gamma=fe_configs.layer_wise_lr_decay_gamma,
                                                          weight_decay=fe_configs.optimizer_weight_decay),
                lr=learning_rates[ModelType.FEATURE_EXTRACTOR],
                weight_decay=fe_configs.optimizer_weight_decay)

            logging.warning(f"Feature Extractor optimizer learning rate is set to {fe_optimizer.param_groups[0]['lr']}")
        else:
            raise NotImplementedError

        if cls_configs.optimizer_type == AdamW:
            cls_optimizer = AdamW(cls.parameters(),
                                  lr=learning_rates[ModelType.CLASSIFIER],
                                  weight_decay=cls_configs.optimizer_weight_decay)
            logging.warning(f"Classifier optimizer learning rate is set to {cls_optimizer.param_groups[0]['lr']}")
        else:
            raise NotImplementedError

        if fe_configs.scheduler_type == ReduceLROnPlateau:
            fe_scheduler = ReduceLROnPlateau(fe_optimizer,
                                             mode=fe_configs.scheduler_mode,
                                             patience=fe_configs.scheduler_patience,
                                             factor=fe_configs.scheduler_factor,
                                             min_lr=fe_configs.scheduler_min_learning_rate,
                                             cooldown=fe_configs.scheduler_cooldown)
        elif fe_configs.scheduler_type == CosineAnnealingLR:
            fe_scheduler = CosineAnnealingLR(
                fe_optimizer,
                T_max=fe_configs.scheduler_cycle_epochs,
                eta_min=fe_configs.scheduler_min_learning_rate,
            )
        else:
            raise NotImplementedError

        if cls_configs.scheduler_type == ReduceLROnPlateau:
            cls_scheduler = ReduceLROnPlateau(cls_optimizer, mode=cls_configs.scheduler_mode,
                                              patience=cls_configs.scheduler_patience,
                                              factor=cls_configs.scheduler_factor,
                                              min_lr=cls_configs.scheduler_min_learning_rate,
                                              cooldown=cls_configs.scheduler_cooldown)
        elif cls_configs.scheduler_type == CosineAnnealingLR:
            cls_scheduler = CosineAnnealingLR(
                cls_optimizer,
                T_max=cls_configs.scheduler_cycle_epochs,
                eta_min=cls_configs.scheduler_min_learning_rate,
            )
        else:
            raise NotImplementedError

        self._optimizers: Dict[ModelType, Optimizer] = {ModelType.FEATURE_EXTRACTOR: fe_optimizer,
                                                        ModelType.CLASSIFIER: cls_optimizer}

        self._schedulers: Dict[ModelType, LRScheduler] = {ModelType.FEATURE_EXTRACTOR: fe_scheduler,
                                                          ModelType.CLASSIFIER: cls_scheduler}

    def set_learning_rate(self, model_type: ModelType, new_base_lr: float) -> None:
        opt = self.get_optimizer(model_type)
        sch = self.get_scheduler(model_type)

        old_base = float(self.learning_rates[model_type])
        new_base = float(new_base_lr)

        scales = []
        for pg in opt.param_groups:
            prev = float(pg.get("lr", old_base))
            scale = (prev / old_base) if old_base > 0 else 1.0
            scales.append(scale)

        new_group_lrs = [new_base * s for s in scales]
        for pg, lr in zip(opt.param_groups, new_group_lrs):
            pg["lr"] = lr

        if hasattr(sch, "base_lrs"):
            sch.base_lrs = list(new_group_lrs)

        self.learning_rates[model_type] = new_base
        logging.warning(
            f"Learning rate for {model_type.name} is set to {new_base} - Learning Rates: {[pg['lr'] for pg in opt.param_groups]} - Weight Decays: {[pg.get('weight_decay', None) for pg in opt.param_groups]}")

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "dropouts": self.dropouts,
            "learning_rates": self.learning_rates
        }

    def save(self, state_path: StoragePath) -> None:
        FileUtil().create_directory(state_path)

        for model_type in self.keys:
            model_path = state_path.join(f'{str(model_type)}_model.pt')
            optimizer_path = state_path.join(f'{str(model_type)}_optimizer.pt')
            scheduler_path = state_path.join(f'{str(model_type)}_scheduler.pt')

            self.get_model(model_type).save(model_path)

            FileUtil().dump(self.get_optimizer(model_type).state_dict(), optimizer_path)
            FileUtil().dump(self.get_scheduler(model_type).state_dict(), scheduler_path)

    def load(self, state_path: StoragePath, weights_only: bool) -> None:
        for model_type in self.keys:
            self._models[model_type].load(state_path.join(f'{str(model_type)}_model.pt'))

            if not weights_only:
                optimizer_state = FileUtil().load(state_path.join(f'{str(model_type)}_optimizer.pt'))
                scheduler_state = FileUtil().load(state_path.join(f'{str(model_type)}_scheduler.pt'))

                self.get_optimizer(model_type).load_state_dict(optimizer_state)
                self.get_scheduler(model_type).load_state_dict(scheduler_state)
                logging.info(f"Loaded {str(model_type)} model, optimizer, and scheduler from {state_path}")
            else:
                logging.info(f"Loaded {str(model_type)} model from {state_path}")

    def get_model(self, model_type: ModelType) -> nn.Module | None:
        if model_type in self._models:
            return self._models[model_type]

        return None

    def get_optimizer(self, model_type: ModelType) -> Optimizer:
        return self._optimizers[model_type]

    def get_scheduler(self, model_type: ModelType) -> LRScheduler:
        return self._schedulers[model_type]

    @property
    def keys(self) -> KeysView[ModelType]:
        return self._models.keys()

    def train(self):
        for key in self._models:
            self._models[key].train()

    def eval(self):
        for key in self._models:
            self._models[key].eval()
