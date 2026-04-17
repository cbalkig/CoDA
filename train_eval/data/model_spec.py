import re
from dataclasses import dataclass

import timm


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_type: str
    image_size: int

    def __init__(self, model_name: str):
        object.__setattr__(self, 'model_name', model_name)
        pattern = re.compile(r'(\w+?)_.*_(\d{3})\.')
        match = pattern.search(model_name)
        if match:
            model_type = match.group(1)
            image_size = int(match.group(2))
        else:
            model_type = model_name.split(".", 1)[0]
            image_size = self.resolve_data_config(model_name=model_name)

        object.__setattr__(self, 'model_type', model_type)
        object.__setattr__(self, 'image_size', image_size)

    @staticmethod
    def resolve_data_config(model_name: str) -> int:
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

        pretrained_cfg = model.pretrained_cfg
        C, H, W = pretrained_cfg['input_size']
        return H
