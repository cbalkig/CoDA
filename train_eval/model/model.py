import inspect
import logging
import pickle
from typing import Dict, KeysView

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from data.file.path import StoragePath
from data.types.domain_type import DomainType
from data.types.model_type import ModelType
from model.managers.dropout_manager import DropoutType
from model.submodel import SubModel
from util.device_detector import DeviceDetector
from util.file_util import FileUtil


class Model:
    def __init__(self, model_folder_path: StoragePath, num_classes: int):
        super().__init__()

        self.model_folder_path = model_folder_path
        self.number_of_classes = num_classes

        self._models: Dict[DomainType, SubModel] = {}

    def save(self, state_path: StoragePath) -> None:
        for domain in self.keys:
            file_path: StoragePath = state_path.join(f'{domain.value}.pkl')
            file_path.path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path.path, "wb") as f:
                pickle.dump(self.get_models(domain).get_state_dict(), f)

            self.get_models(domain).save(state_path.join(str(domain)))

    def load(self, state_path: StoragePath, weights_only: bool = False) -> None:
        for domain in DomainType:
            path = state_path.join(str(domain))
            if FileUtil().exists(path):
                if domain not in self.keys:
                    file_path: StoragePath = state_path.join(f'{domain.value}.pkl')
                    with open(file_path.path, "rb") as f:
                        params = pickle.load(f)

                    allowed = set(inspect.signature(SubModel).parameters.keys())
                    params = {k: v for k, v in params.items() if k in allowed}

                    self.add_model(domain, SubModel(**params))

                self.get_models(domain).load(path, weights_only)
                logging.warning(f'Loaded {domain} model - {path}')
            else:
                logging.debug(f"Model for domain {domain} not found at {path}")

    def copy_model(self, from_domain: DomainType, to_domain: DomainType) -> None:
        for model in self.get_models(from_domain).keys:
            self.get_model(to_domain, model).load_state_dict(self.get_model(from_domain, model).state_dict())
            logging.warning(f"Copied {from_domain}-{model} model to {to_domain}-{model}")

    def train_mode(self) -> None:
        self.eval_mode()

        for domain in self.keys:
            for model in self.get_models(domain).keys:
                self.get_model(domain, model).train()

    def eval_mode(self):
        for domain in self.keys:
            for model_type in self.get_models(domain).keys:
                self.get_model(domain, model_type).eval()

    def get_learning_rates(self) -> Dict[str, float]:
        lr: Dict[str, float] = {}

        for domain in self.keys:
            sub = self.get_models(domain)
            for model_type in sub.keys:
                opt = sub.get_optimizer(model_type)
                sch = sub.get_scheduler(model_type)

                value: float
                # 1) Prefer the actually applied per-group LRs on the optimizer
                try:
                    group_lrs = [float(pg.get("lr", 0.0)) for pg in opt.param_groups]
                    if group_lrs:
                        value = sum(group_lrs) / len(group_lrs)  # mean over groups
                    else:
                        raise ValueError("No param groups.")
                except Exception:
                    # 2) Fallback to scheduler-reported last LRs (also per-group)
                    try:
                        get_last_lr = getattr(sch, "get_last_lr", None)
                        if callable(get_last_lr):
                            last = [float(x) for x in sch.get_last_lr()]
                            value = (sum(last) / len(last)) if last else float(opt.param_groups[0].get("lr", 0.0))
                        else:
                            value = float(opt.param_groups[0].get("lr", 0.0))
                    except Exception:
                        # 3) Last resort: first param group LR or 0.0
                        value = float(opt.param_groups[0].get("lr", 0.0)) if opt.param_groups else 0.0

                lr[f"{domain.value}_{model_type.value}"] = value

        return lr

    def get_dropouts(self) -> Dict[str, float]:
        dropouts: Dict[str, float] = {}

        for domain in self.keys:
            sub = self.get_models(domain)
            for model_type in sub.keys:
                for dropout_type in DropoutType:
                    try:
                        val = self.get_model(domain, model_type).get_dropout(dropout_type)
                    except NotImplementedError:
                        continue
                    if val is None:
                        continue
                    dropouts[f"{domain.value}_{model_type.value}_{dropout_type.value}"] = float(val)
        return dropouts

    def get_num_layers(self) -> Dict[str, float]:
        num_layers: Dict[str, float] = {}

        for domain in self.keys:
            sub = self.get_models(domain)
            for model_type in sub.keys:
                if model_type != ModelType.FEATURE_EXTRACTOR:
                    continue

                num_layers[f"{domain.value}_{model_type.value}"] = self.get_model(domain,
                                                                                  model_type).number_of_layers

                num_layers[f"{domain.value}_{model_type.value}_unfrozen"] = self.get_model(domain,
                                                                                           model_type).number_of_unfrozen_layers

        return num_layers

    def add_model(self, domain: DomainType, submodel: SubModel) -> None:
        if not domain in self.keys:
            self._models[domain] = submodel
            logging.warning(f"Added {domain} model")

    def remove_models(self, domain: DomainType) -> None:
        if domain in self.keys:
            del self._models[domain]
            logging.warning(f"Removed {domain} model")

        DeviceDetector().empty_cache()

    def get_model(self, domain: DomainType, model_type: ModelType) -> nn.Module | None:
        model = self.get_models(domain)
        if not model:
            return None

        return model.get_model(model_type)

    def get_models(self, domain: DomainType) -> SubModel | None:
        return self._models[domain] if domain in self.keys else None

    def get_optimizer(self, domain: DomainType, model_type: ModelType) -> Optimizer:
        return self.get_models(domain).get_optimizer(model_type)

    def get_scheduler(self, domain: DomainType, model_type: ModelType) -> LRScheduler:
        return self.get_models(domain).get_scheduler(model_type)

    @property
    def keys(self) -> KeysView[DomainType]:
        return self._models.keys()

    @staticmethod
    def _get_lr_safe(scheduler: LRScheduler, optimizer: Optimizer) -> float:
        if hasattr(scheduler, "get_last_lr"):
            vals = scheduler.get_last_lr()
            if isinstance(vals, (list, tuple)):
                return float(vals[0])

        return float(optimizer.param_groups[0]["lr"])
