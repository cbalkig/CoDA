import importlib
import io
from typing import Tuple, List, Dict, Any, Union

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.preprocessing import LabelEncoder

from configs.base.configs import Configs
from data.dataset.abstract import AbstractDataset
from data.file.path import StoragePath
from util.file_util import FileUtil


class FileDataset(AbstractDataset):
    def __init__(
            self,
            paths: Union[List[StoragePath], Dict[int, List[StoragePath]]],
            data_augment: bool,
            num_classes: int,
            label_encoder: LabelEncoder
    ) -> None:
        super().__init__(data_augment=data_augment, num_classes=num_classes)
        self._label_encoder = label_encoder

        if isinstance(paths, dict):
            flat_paths: List[StoragePath] = []
            flat_labels: List[int] = []

            for y, plist in paths.items():
                if not isinstance(plist, list):
                    raise TypeError(
                        f"Dict values must be List[StoragePath], got {type(plist)} for label '{y}'"
                    )

                for p in plist:
                    flat_paths.append(p)
                    flat_labels.append(y)

            self._paths = flat_paths
            self._labels = flat_labels
        if isinstance(paths, list):
            self._paths = paths
            self._labels = [self._encode_label_from_path(p) for p in self._paths]

        self.file_pattern = Configs().dataset.dataset_type.file_pattern if Configs().dataset.dataset_type is not None else None

        data_augment_modules = Configs().dataset.data_augmentations
        if data_augment_modules is None:
            raise Exception(f'Data augmentation modules not defined.')

        self.transforms: Dict[str, Any] = {}
        for key in data_augment_modules.keys():
            data_augment_module = data_augment_modules[key]
            module = importlib.import_module(f"data.variations.data_augmentation.{data_augment_module}")
            TransformClass = getattr(module, "DataTransforms")
            self.transforms[key] = TransformClass(
                image_size=Configs().feature_extractor.timm_model.image_size
            )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, int, str]:
        path = self._paths[idx]

        try:
            img = Image.open(io.BytesIO(FileUtil().read_file(path)))
        except UnidentifiedImageError:
            raise UnidentifiedImageError(f"File {path} is not a valid image.")

        img = ImageOps.exif_transpose(img)  # fix orientation

        # Always ensure RGB without alpha
        if img.mode in ('RGBA', 'LA', 'P'):
            # Convert with background to remove transparency
            background = Image.new("RGB", img.size, (255, 255, 255))  # white background
            img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1])  # paste without alpha
            img = background
        else:
            img = img.convert('RGB')

        if self.file_pattern is not None:
            file_type = self.file_pattern.match(path.path.name)
            if file_type is None or file_type.groups()[0] is None:
                prefix = 'default'
            else:
                prefix = file_type.groups()[0]
        else:
            prefix = 'default'

        if self.data_augment:
            img = self.transforms[prefix].train_transforms()(img)
        else:
            img = self.transforms[prefix].test_transforms()(img)

        label = self._labels[idx]

        return idx, img, label, str(path)

    def _encode_label_from_path(self, path: StoragePath) -> int:
        return int(self._label_encoder.transform([path.parent_name])[0])

    def get_labels(self) -> torch.Tensor:
        return torch.tensor(self._labels)

    def __add__(self, other: 'FileDataset') -> 'FileDataset':
        if not isinstance(other, FileDataset):
            raise TypeError(f"unsupported operand type(s) for +: 'FileDataset' and '{type(other)}'")

        if self.num_classes != other.num_classes:
            raise ValueError("Datasets must have the same number of classes to be merged.")

        combined_paths = self._paths + other._paths

        return FileDataset(
            paths=combined_paths,
            data_augment=self.data_augment,
            num_classes=self.num_classes,
            label_encoder=self._label_encoder
        )
