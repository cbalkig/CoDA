from typing import Dict

from configs.base.section import Section


class PretrainedConfig:
    def __init__(self, cfg: Section):
        self.weights_only: bool = False if cfg.get('weights_only') is None else cfg.getboolean('weights_only')
        self.model: str = cfg.get('model')
        self.base_models: Dict[int | None, str] = {}

        base_models = cfg.get('base_models')
        if base_models is not None:
            for raw_key in base_models.keys():
                key = str(raw_key).strip()
                val = base_models[raw_key].strip()

                if key.lower() in {"none", "null", "nil", ""}:
                    self.base_models[None] = val
                else:
                    try:
                        self.base_models[int(key)] = val
                    except ValueError as e:
                        raise ValueError(
                            f"[base_models] has invalid key {raw_key!r}; expected an integer or 'None'."
                        ) from e
