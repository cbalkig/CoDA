import csv
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import math
import pandas as pd
import seaborn as sns
import torch
from PIL.Image import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, \
    recall_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from configs.base.configs import Configs
from data.data_tag import DataTag
from data.dataset.data import Data
from data.file.path import StoragePath
from data.loss.loss import Loss
from data.loss.losses import Losses
from data.metrics import Metrics
from data.model_tag import ModelTag
from data.stages.epoch_stages.init_stage import EpochInitStage
from data.stages.epoch_stages.train_stage import EpochTrainStage
from data.stages.stage_base import StageBase
from data.stages.stage_types import Stages
from data.stats.stats import Stats
from data.types.data_type import DataType
from data.types.domain_type import DomainType
from data.types.model_type import ModelType
from model.grad_cam import GradCAM
from model.managers.dropout_manager import DropoutManager, DropoutType
from model.managers.mixup_manager import MixupManager
from model.managers.pseudo_label_manager import PseudoLabelManager
from model.model import Model
from model.submodel import SubModel
from util.device_detector import DeviceDetector
from util.file_util import FileUtil


class EvaluateModelTypeStage(StageBase):
    def __init__(self) -> None:
        super().__init__(Stages.EVAL_MODELS, False, None, eval=True)

        self.current_model: Optional[str] = None
        self.current_epoch: Optional[int] = None
        self.report_file: Path = Configs().evaluation.report_folder / f'{Configs().general.tag}.csv'

    def get_improvement(self, losses: Dict[ModelTag, Loss], f1s: Dict[ModelTag, float]) -> Tuple[
        Dict[ModelTag, float], Dict[ModelTag, float]]:
        return {}, {}

    def get_action_completed(self) -> bool:
        return True

    def preprocess(self, epoch: int, model: Model, data: Data, dropout_manager: DropoutManager,
                   mixup_manager: MixupManager, pseudo_label_manager: PseudoLabelManager) -> None:
        dropout_manager.reset()
        mixup_manager.reset()

        if DomainType.SOURCE not in model.keys:
            model.add_model(DomainType.SOURCE, SubModel(model.number_of_classes, dropouts={
                ModelType.FEATURE_EXTRACTOR: {
                    DropoutType.DROPOUT: Configs().feature_extractor.drop_rate(self.stage),
                    DropoutType.DROPOUT_PATH: Configs().feature_extractor.drop_path_rate(self.stage)
                },
                ModelType.CLASSIFIER: {
                    DropoutType.DROPOUT: Configs().classifier.drop_rate(self.stage)
                },
            }, learning_rates={
                ModelType.FEATURE_EXTRACTOR: Configs().feature_extractor.optimizer_learning_rate(self.stage),
                ModelType.CLASSIFIER: Configs().classifier.optimizer_learning_rate(self.stage)
            }))

    def _write_report(self, record: Dict[str, Any]) -> None:
        current_df = pd.DataFrame.from_records([record])

        if not current_df.empty:
            current_df = current_df.groupby(['Model', 'Source Cross Val K', 'Target Cross Val K'],
                                            as_index=False).first()

        if self.report_file.exists():
            existing_df = pd.read_csv(self.report_file)
            existing_df = existing_df[existing_df['Source Cross Val K'] != 'Average']
        else:
            existing_df = pd.DataFrame()

        full_df = pd.concat([existing_df, current_df], ignore_index=True)
        full_df['Source Cross Val K'] = full_df['Source Cross Val K'].astype(int)
        full_df['Target Cross Val K'] = full_df['Target Cross Val K'].astype(int)
        full_df = full_df.groupby(['Model', 'Source Cross Val K', 'Target Cross Val K'], as_index=False).first()
        epoch_cols = [col for col in full_df.columns if col.endswith(' Epoch')]
        f1_cols = [col for col in full_df.columns if col.endswith(' F1')]

        for col in f1_cols + epoch_cols:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

        df_avg = full_df.groupby('Model', as_index=False)[f1_cols + epoch_cols].mean()
        df_avg['Source Cross Val K'] = 'Average'
        df_avg['Target Cross Val K'] = 'Average'

        for col in epoch_cols:
            if col in full_df.columns:
                full_df[col] = full_df[col].round().astype('Int64')
            if col in df_avg.columns:
                df_avg[col] = df_avg[col].round().astype('Int64')

        def strict_ceil_precision(x):
            if pd.isna(x):
                return x
            if x > 0.5:
                return math.ceil(x * 100) / 100
            else:
                return round(x, 2)

        for col in f1_cols:
            if col in df_avg.columns:
                df_avg[col] = df_avg[col].apply(strict_ceil_precision)

        final_output = pd.concat([full_df, df_avg], ignore_index=True)

        if hasattr(self.report_file, 'parent'):
            self.report_file.parent.mkdir(parents=True, exist_ok=True)

        final_output.to_csv(self.report_file, index=False)

    def run(self, train_stage: EpochTrainStage) -> None:
        source_train_tag = DataTag(DomainType.SOURCE, DataType.TRAIN)
        source_val_tag = DataTag(DomainType.SOURCE, DataType.VALIDATION)
        source_test_tag = DataTag(DomainType.SOURCE, DataType.TEST)

        target_train_tag = DataTag(DomainType.TARGET, DataType.TRAIN)
        target_val_tag = DataTag(DomainType.TARGET, DataType.VALIDATION)
        target_test_tag = DataTag(DomainType.TARGET, DataType.TEST)

        train_stage.data.sample_evaluation([DomainType.SOURCE, DomainType.TARGET])

        source_test_tags = []
        target_test_tags = []
        for test_tag in Configs().storage.test_folders:
            source_test_tags.append(DataTag(DomainType.SOURCE, DataType.TEST, identifier=test_tag))
            target_test_tags.append(DataTag(DomainType.TARGET, DataType.TEST, identifier=test_tag))

        source_train_loaders = train_stage.data.get_evaluation_dataloaders(source_train_tag)
        source_val_loaders = train_stage.data.get_evaluation_dataloaders(source_val_tag)
        source_test_loaders = train_stage.data.get_evaluation_dataloaders(source_test_tag)
        target_train_loaders = train_stage.data.get_evaluation_dataloaders(target_train_tag)
        target_val_loaders = train_stage.data.get_evaluation_dataloaders(target_val_tag)
        target_test_loaders = train_stage.data.get_evaluation_dataloaders(target_test_tag)

        source_train_on_source_tag = ModelTag(source_train_tag, DomainType.SOURCE)
        source_val_on_source_tag = ModelTag(source_val_tag, DomainType.SOURCE)
        target_train_on_target_tag = ModelTag(target_train_tag, DomainType.TARGET)
        target_val_on_target_tag = ModelTag(target_val_tag, DomainType.TARGET)

        if source_train_loaders is not None and len(source_train_loaders) > 0:
            self._evaluate_and_report(train_stage.model, train_stage.data, source_train_on_source_tag,
                                      source_train_loaders[0], 'Source Train')

        if source_val_loaders is not None and len(source_val_loaders) > 0:
            self._evaluate_and_report(train_stage.model, train_stage.data, source_val_on_source_tag,
                                      source_val_loaders[0], 'Source Validation')

        if source_test_loaders is not None and len(source_test_loaders) > 0:
            self._evaluate_and_report(train_stage.model, train_stage.data, source_train_on_source_tag,
                                      source_test_loaders[0],
                                      'Source Test on Source Train')

            self._evaluate_and_report(train_stage.model, train_stage.data, source_val_on_source_tag,
                                      source_test_loaders[0],
                                      'Source Test on Source Validation')
        else:
            logging.info("No test data on source domain. Skipping evaluation.")

        if len(source_test_tags) == 0:
            logging.info("No data on test domain. Skipping evaluation.")
        else:
            for test_tag in source_test_tags:
                test_loaders = train_stage.data.get_evaluation_dataloaders(test_tag)
                if test_loaders is not None and len(test_loaders) > 0:
                    self._evaluate_and_report(train_stage.model, train_stage.data, source_train_on_source_tag,
                                              test_loaders[0],
                                              f'Target ({test_tag.identifier}) on Source Train' if test_tag.identifier is not None else f'Target (None) on Source Train')

                    self._evaluate_and_report(train_stage.model, train_stage.data, source_val_on_source_tag,
                                              test_loaders[0],
                                              f'Target ({test_tag.identifier}) on Source Validation' if test_tag.identifier is not None else f'Target (None) on Source Validation')

        if target_train_loaders is not None and len(target_train_loaders) > 0:
            self._evaluate_and_report(train_stage.model, train_stage.data, source_val_on_source_tag,
                                      target_train_loaders[0], 'Target Train on Source Validation', pseudo_label=True)

        if target_train_loaders is not None and len(target_train_loaders) > 0:
            self._evaluate_and_report(train_stage.model, train_stage.data, target_train_on_target_tag,
                                      target_train_loaders[0], 'Target Train')

        if target_val_loaders is not None and len(target_val_loaders) > 0:
            self._evaluate_and_report(train_stage.model, train_stage.data, target_val_on_target_tag,
                                      target_val_loaders[0], 'Target Validation')

        if target_test_loaders is not None and len(target_test_loaders) > 0:
            self._evaluate_and_report(train_stage.model, train_stage.data, target_train_on_target_tag,
                                      target_test_loaders[0], 'Target Test on Target Train')

            self._evaluate_and_report(train_stage.model, train_stage.data, target_val_on_target_tag,
                                      target_test_loaders[0], 'Target Test on Target Validation')
        else:
            logging.info("No test data on target domain. Skipping evaluation.")

        if len(target_test_tags) == 0:
            logging.info("No custom test data on target domain. Skipping evaluation.")
        else:
            for test_tag in target_test_tags:
                test_loaders = train_stage.data.get_evaluation_dataloaders(test_tag)
                if test_loaders is not None and len(test_loaders) > 0:
                    self._evaluate_and_report(train_stage.model, train_stage.data, target_train_on_target_tag,
                                              test_loaders[0],
                                              f'Target ({test_tag.identifier}) on Target Train' if test_tag.identifier is not None else f'Target (None) on Target Train', )

                    self._evaluate_and_report(train_stage.model, train_stage.data, target_val_on_target_tag,
                                              test_loaders[0],
                                              f'Target ({test_tag.identifier}) on Target Validation' if test_tag.identifier is not None else f'Target (None) on Target Validation', )

    def step_optimizers(self, model: Model, loss: Loss) -> None:
        pass

    def get_train_models(self, model: Model) -> List[nn.Module]:
        pass

    def step_schedulers(self, model: Model, stats: Stats) -> None:
        pass

    def init_epoch(self, init_stage: EpochInitStage) -> None:
        pass

    def get_optimizers(self, model: Model) -> List[Optimizer]:
        pass

    def _evaluate(self, fe: nn.Module, clf: nn.Module, dataset_name: str, reports_path: StoragePath, data: Data,
                  loader: DataLoader, model_tag: ModelTag) -> tuple[list[Any], list[Any], list[Any],
    list[Any], list[Any], list[Any], list[Any]] | None:
        if loader is None:
            return None

        paths = []
        images = []
        features = []
        labels = []
        predictions = []
        max_probabilities = []
        probabilities = []

        for batch in loader:
            _images = batch[1].to(DeviceDetector().device)
            _labels = batch[2].to(DeviceDetector().device)
            _paths = batch[3]

            images.extend([img.cpu() for img in _images])
            with torch.no_grad():
                _features = fe(_images)
                _outputs = clf(_features)
                _probs = torch.softmax(_outputs, dim=1)
                _max_probs, _preds = torch.max(_probs, 1)

                features.append(_features.cpu())
                predictions.extend(_preds.cpu().tolist())
                probabilities.extend(_probs.cpu().numpy())
                max_probabilities.extend(_max_probs.cpu().numpy())
                labels.extend(_labels.cpu().numpy())
                paths.extend(list(_paths))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.mps.is_available():
            torch.mps.synchronize()

        # Calculate Metrics
        metrics = Metrics(
            accuracy_score(labels, predictions),
            f1_score(labels, predictions, average="macro", zero_division=0),
            precision_score(labels, predictions, average="macro", zero_division=0),
            recall_score(labels, predictions, average="macro", zero_division=0),
            classification_report(labels, predictions, output_dict=True, zero_division=0),
            confusion_matrix(labels, predictions, labels=range(len(data.label_encoder.classes_)))
        )

        logging.warning(
            f'Evaluation on - {dataset_name} - {model_tag.tag} - F1: {(metrics.f1 * 100):.2f}% (Epoch: {self.current_epoch}) (Number of images: {len(images)})')

        self._write_report({
            'Model': Configs().general.tag,
            f'{dataset_name} Epoch': self.current_epoch,
            f'{dataset_name} F1': f'{(metrics.f1 * 100):.2f}',
            'Source Cross Val K': data.source_cross_val_k if data.source_cross_val_k is not None else -1,
            'Target Cross Val K': data.target_cross_val_k if data.target_cross_val_k is not None else -1,
        })

        # Save Metrics to Text File
        file = reports_path.join("metrics.txt")
        with open(file.path, 'w') as f:
            f.write(f"Evaluation on - {model_tag.tag}\n")
            f.write("Model Evaluation Metrics\n")
            f.write("=========================\n\n")

            f.write(f"Accuracy          : {metrics.acc:.4f}\n")
            f.write(f"F1-score (macro)  : {metrics.f1:.4f}\n")
            f.write(f"Precision (macro) : {metrics.precision:.4f}\n")
            f.write(f"Recall (macro)    : {metrics.recall:.4f}\n\n")

            f.write("Classification Report\n")
            f.write("----------------------\n")
            headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
            f.write(f"{headers[0]:<15}{headers[1]:>10}{headers[2]:>10}{headers[3]:>10}{headers[4]:>10}\n")
            f.write("-" * 55 + "\n")

            for label, m in metrics.cr.items():
                if isinstance(m, dict):
                    f.write(f"{label:<15}"
                            f"{m['precision']:>10.4f}"
                            f"{m['recall']:>10.4f}"
                            f"{m['f1-score']:>10.4f}"
                            f"{m['support']:>10}\n")

            f.write("\n")

            class_labels = data.label_encoder.classes_.tolist()
            f.write("Confusion Matrix\n")
            f.write("----------------\n")
            f.write(f"{'':12}" + ''.join(f"{label:>12}" for label in class_labels) + "\n")

            for label, row in zip(class_labels, metrics.cm):
                f.write(f"{label:12}")
                f.write(''.join(f"{val:12}" for val in row))
                f.write("\n")

        return paths, images, features, labels, predictions, probabilities, max_probabilities

    @staticmethod
    def _save_one_blend_and_copy(blended_pil: Image, blended_path: StoragePath, src_image_path: StoragePath,
                                 dst_dir: StoragePath):
        try:
            parent = blended_path.path.parent if hasattr(blended_path, "path") else blended_path.parent
            if hasattr(blended_path, "local") and blended_path.local:
                parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        actual_path = blended_path.path if hasattr(blended_path, "path") else blended_path
        save_path = actual_path.with_suffix(".png")
        blended_pil.save(save_path)

        FileUtil().copy_file(src_image_path, dst_dir.join(src_image_path.name))

    def _load_model(self, model: Model, data: Data, model_tag: ModelTag, dataset_name: str) -> Any:
        if self.current_model == model_tag.best_model_tag:
            return None

        model_folder_path: StoragePath = data.model_path
        path = model_folder_path.join(model_tag.best_model_tag)
        if not FileUtil().exists(path):
            if Configs().feature_extractor.pretrained:
                logging.warning(f'Model {dataset_name} loaded from pretrained model.')
                return None
            else:
                logging.warning(f'Skipping {dataset_name} as no model exists - {path}')
                return None
        else:
            epoch_data = FileUtil().load(path.join(f'epoch_data.pkl'))
            logging.info(f'Model recovered from Epoch: {epoch_data.get("epoch")} - {path}')

            model.load(path, weights_only=True)
            self.current_model = model_tag.best_model_tag
            self.current_epoch = epoch_data.get("epoch")
            return epoch_data

    def _evaluate_and_report(self, model: Model, data: Data, model_tag: ModelTag, loader: DataLoader,
                             dataset_name: str, pseudo_label: bool = False) -> None:
        if model is None:
            logging.info(f'Skipping {dataset_name} as no model')
            return None

        if self._already_executed(data.source_cross_val_k,
                                  data.target_cross_val_k if data.target_cross_val_k is not None else -1, dataset_name):
            return None

        logging.info(f'Evaluating ---> {dataset_name} on ---> {model_tag.tag}')
        _ = self._load_model(model, data, model_tag, dataset_name)

        reports_path: StoragePath = StoragePath(
            Configs().evaluation.report_folder / 'reports' / f'{Configs().pretrained.base_models[data.source_cross_val_k]}' / dataset_name.lower().replace(
                " ",
                "_").replace(
                "(",
                "").replace(
                ")", ""))

        logging.info(f'Report directory - {reports_path}')

        if reports_path.local:
            FileUtil().delete_directory(reports_path)
            reports_path.path.mkdir(parents=True, exist_ok=True)

        if loader is None:
            logging.debug(f'Skipping {dataset_name} as no data loaded')
            return None

        logging.debug(f'Loader size: {len(loader)}')
        logging.debug(f'Data size: {len(loader.dataset)}')

        device = DeviceDetector().device
        model.eval_mode()

        fe = model.get_model(model_tag.data_tag.domain, ModelType.FEATURE_EXTRACTOR)
        if fe is None:
            logging.info(f'Skipping {dataset_name} as no feature extractor model')
            return None

        clf = model.get_model(model_tag.data_tag.domain, ModelType.CLASSIFIER)

        fe.eval()
        clf.eval()

        # Run evaluation
        paths, images, features, labels, predictions, probabilities, max_probabilities = self._evaluate(
            fe, clf, dataset_name, reports_path, data, loader, model_tag
        )

        class_names = data.label_encoder.classes_.tolist()

        if Configs().evaluation.reports:
            records = []
            correct_dir = reports_path.join('images', 'correct')
            incorrect_dir = reports_path.join('images', 'incorrect')

            cam_dir = reports_path.join("grad_cams")
            if cam_dir.local:
                cam_dir.path.mkdir(exist_ok=True)

            combined = nn.Sequential(fe, clf).to(device)
            combined.eval()
            cam_extractor = GradCAM(combined)

            io_pool = ThreadPoolExecutor(max_workers=32)

            for i, (pred_idx, true_label_idx, image_path) in enumerate(zip(predictions, labels, paths)):
                image_path = StoragePath(image_path)
                label_true = class_names[int(true_label_idx)]
                label_pred = class_names[int(pred_idx)]

                p_pred = float(max_probabilities[i])
                p_true = float(probabilities[i][int(true_label_idx)])

                bin_str = f"{math.floor(p_pred * 10) / 10:.2f}"
                x = images[i].unsqueeze(0).to(device)

                torch.set_grad_enabled(True)
                blended = cam_extractor(x, int(true_label_idx))
                torch.set_grad_enabled(False)

                if label_pred != label_true:
                    blended_file_path = cam_dir.join('incorrect', f'{label_true}_as_{label_pred}',
                                                     bin_str, image_path.path.with_suffix('.png').name)
                    save_path = incorrect_dir.join(f'{label_true}_as_{label_pred}', bin_str)
                    correct_flag = 0
                else:
                    blended_file_path = cam_dir.join('correct', f'{label_true}', bin_str,
                                                     image_path.path.with_suffix('.png').name)
                    save_path = correct_dir.join(f'{label_true}', bin_str)
                    correct_flag = 1

                if blended_file_path.local:
                    blended_file_path.path.parent.mkdir(parents=True, exist_ok=True)

                io_pool.submit(
                    self._save_one_blend_and_copy,
                    blended,
                    blended_file_path,
                    image_path,
                    save_path
                )

                prob_vector = probabilities[i]

                mu = prob_vector.mean()
                sigma = prob_vector.std(ddof=0) if prob_vector.std(ddof=0) > 0 else 1e-8
                z_scores = (prob_vector - mu) / sigma

                record = {
                    "path": str(image_path),
                    "true_idx": int(true_label_idx),
                    "true_name": label_true,
                    "pred_idx": int(pred_idx),
                    "pred_name": label_pred,
                    "correct": correct_flag,
                    "pred_prob": p_pred,
                    "true_prob": p_true,
                    "bin": bin_str,
                }

                for cname, p, z in zip(class_names, prob_vector, z_scores):
                    record[f"prob_{cname}"] = float(p)
                    record[f"z_{cname}"] = float(z)
                records.append(record)

            io_pool.shutdown(wait=True)

            csv_path = Path(reports_path.join("predictions.csv").path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if records:
                header = list(records[0].keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(records)
            else:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f,
                                            fieldnames=["path", "true_idx", "true_name", "pred_idx", "pred_name",
                                                        "correct",
                                                        "pred_prob", "true_prob", "bin"])
                    writer.writeheader()

            file = reports_path.join("cm.png")
            cm = confusion_matrix(labels, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(file.path)
            plt.close()

            file = reports_path.join('classification_report.csv')
            cr = classification_report(labels, predictions, target_names=class_names, output_dict=True, zero_division=0)
            cr_df = pd.DataFrame(cr).transpose()
            cr_df.to_csv(file.path)

            file = reports_path.join('summary.csv')
            summary_df = pd.DataFrame({"True": labels, "Predicted": predictions})
            summary_df = pd.crosstab(summary_df["True"], summary_df["Predicted"], rownames=["Actual"],
                                     colnames=["Predicted"], margins=True)
            summary_df.to_csv(file.path)

        if pseudo_label and Configs().evaluation.pseudo_labels:
            logging.info('Generating Pseudo Label Analysis...')

            pl_dir = reports_path.join("pseudo_labels")

            pl_gradcams_root = pl_dir.join("gradcams")
            pl_images_root = pl_dir.join("images")

            if pl_dir.local:
                pl_gradcams_root.path.mkdir(parents=True, exist_ok=True)
                pl_images_root.path.mkdir(parents=True, exist_ok=True)

            # ---  Calculate Thresholds (Ported from train_target.py) ---
            # We map predictions to their max probabilities to calculate dynamic thresholds if needed
            confs_map: Dict[str, List[torch.Tensor]] = defaultdict(list)

            # Re-organize data for threshold calculation
            for pred_idx, max_prob in zip(predictions, max_probabilities):
                label_name = class_names[int(pred_idx)]
                # Convert numpy/float to tensor for consistency with train_target logic
                confs_map[label_name].append(torch.tensor(max_prob))

            pl_type = Configs().pseudo_label.type(self.stage)
            # Note: For eval, we use the initial_threshold from config as the "current" reference
            # since we don't have the specific epoch-based manager state here.
            base_threshold_val = Configs().pseudo_label.initial_threshold(self.stage)

            final_thresholds: Dict[str, float] = {}

            if pl_type == 'dynamic_top':
                # Dynamic Logic: Sort confidences per class and pick top K%
                for c in class_names:
                    conf_list = confs_map.get(c, [])
                    if not conf_list:
                        final_thresholds[c] = 1.1  # Impossible threshold
                        continue

                    conf_tensor = torch.stack(conf_list)
                    sorted_confs, _ = torch.sort(conf_tensor, descending=True)
                    n = len(sorted_confs)
                    k = max(1, int(n * base_threshold_val))

                    # The threshold is the score of the k-th item
                    top_k = sorted_confs[:k]
                    final_thresholds[c] = top_k[-1].item()
                    logging.warning(f"Top-k: {top_k} - Final threshold: {final_thresholds[c]}")
            else:
                # Fixed Logic
                for c in class_names:
                    final_thresholds[c] = base_threshold_val

            # --- Filter and Generate Visuals ---

            # Prepare GradCAM extractor specifically for PL (in case it wasn't initialized above)
            combined = nn.Sequential(fe, clf).to(device)
            combined.eval()
            cam_extractor = GradCAM(combined)

            pl_records = []

            # Re-use ThreadPool if available or run sequentially (running sequentially here for safety/simplicity)
            # We iterate through all evaluated items and check if they pass PL criteria
            for i, (pred_idx, true_label_idx, image_path, max_prob) in enumerate(
                    zip(predictions, labels, paths, max_probabilities)):
                pred_name = class_names[int(pred_idx)]
                true_name = class_names[int(true_label_idx)]

                # Check Threshold
                threshold_to_beat = final_thresholds.get(pred_name, 1.1)

                if max_prob >= threshold_to_beat:
                    # ** This is a Pseudo-Label **

                    # Determine if the Pseudo-Label is actually correct (for debugging purposes)
                    is_actually_correct = (pred_idx == true_label_idx)

                    # Generate Grad-CAM
                    image_tensor = images[i].unsqueeze(0).to(device)
                    torch.set_grad_enabled(True)
                    # We generate CAM for the PREDICTED class (because that's what the pseudo-label is)
                    blended = cam_extractor(image_tensor, int(pred_idx))
                    torch.set_grad_enabled(False)

                    # Format filename: [Prob]_[True]_[Pred].png
                    bin_str = f"{math.floor(max_prob * 100)}pct"
                    fname = f"{bin_str}_{true_name}_as_{pred_name}_{Path(image_path).stem}.png"

                    # --- CHANGE START: Determine Save Paths for both GradCAM and Original Image ---

                    # Define relative folder structure: correct/Class vs incorrect/True_as_Pred
                    if is_actually_correct:
                        sub_path = Path("correct") / pred_name
                    else:
                        sub_path = Path("incorrect") / f"{true_name}_as_{pred_name}"

                    # 1. Setup GradCAM Directory
                    gradcam_save_dir = pl_gradcams_root.join(str(sub_path))
                    if gradcam_save_dir.local:
                        gradcam_save_dir.path.mkdir(parents=True, exist_ok=True)

                    # 2. Setup Image Directory
                    image_save_dir = pl_images_root.join(str(sub_path))
                    if image_save_dir.local:
                        image_save_dir.path.mkdir(parents=True, exist_ok=True)

                    # --- SAVE OPERATIONS ---

                    # Save the blended GradCAM image
                    target_cam_file = gradcam_save_dir.join(fname)
                    actual_cam_path = target_cam_file.path if hasattr(target_cam_file, "path") else target_cam_file
                    blended.save(actual_cam_path)

                    # Save the Original Image (Copy from source)
                    # We use the same 'fname' so you can easily match the original to the GradCAM
                    src_path_obj = StoragePath(image_path)
                    target_image_file = image_save_dir.join(fname)
                    FileUtil().copy_file(src_path_obj, target_image_file)

                    # --- CHANGE END ---

                    # Add to CSV Record
                    pl_records.append({
                        "path": str(image_path),
                        "true_name": true_name,
                        "pred_name": (pred_name + " (PL)"),  # Mark as PL
                        "confidence": max_prob,
                        "threshold_used": threshold_to_beat,
                        "margin": max_prob - threshold_to_beat,
                        "is_correct_pl": 1 if is_actually_correct else 0
                    })

            # --- Save CSV Report ---
            csv_path = Path(pl_dir.join("pseudo_labels.csv").path)
            if pl_records:
                header = list(pl_records[0].keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    writer.writeheader()
                    writer.writerows(pl_records)

            logging.info(f"Pseudo-Label Analysis Complete. Identified {len(pl_records)} samples.")
        return None

    def get_train_status(self) -> Dict[str, float | int]:
        return {}

    def get_total_loss(self, stage: EpochTrainStage) -> Losses:
        return Losses({
        })

    def _already_executed(self, source_cross_val_k: int, target_cross_val_k: int, dataset_name: str) -> bool:
        if not self.report_file.exists():
            return False

        df = pd.read_csv(self.report_file)
        df = df[(df['Model'] == Configs().general.tag) & (df['Source Cross Val K'] == str(source_cross_val_k)) & (
                df['Target Cross Val K'] == str(target_cross_val_k))].copy()

        if len(df) == 0:
            return False

        if f'{dataset_name} F1' not in df.columns:
            return False

        if str(df[f'{dataset_name} F1'].iloc[-1]) != 'nan':
            return True

        return False
