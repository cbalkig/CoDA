import logging
import os
import random

import numpy as np
import torch

from configs.base.configs import Configs
from data.file.path import StoragePath


class HelperFuncs:

    @staticmethod
    def seed_everything() -> None:
        seed = Configs().general.seed

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Prefer speed/stability on Ampere:
        torch.set_float32_matmul_precision("medium")  # leave matmul flexible

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if torch.mps.is_available():
            torch.mps.manual_seed(seed)

        logging.debug("Seeding complete (Ampere-optimized).")

    @staticmethod
    def close_logging():
        # Never shut down logging globally. Preserve console logging.
        # We only remove/close file handlers to safely release files.
        root = logging.getLogger()
        for handler in list(root.handlers):
            try:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    root.removeHandler(handler)
            except Exception:
                logging.debug("Ignoring error while closing a file handler", exc_info=True)
        logging.debug("Logging cleanup done (console handlers preserved).")

    @staticmethod
    def setup_logging() -> None:
        logging.shutdown()

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s"
        )

        console_handler.setFormatter(formatter)

        logging.basicConfig(level=logging.DEBUG, handlers=[console_handler])

    @staticmethod
    def restart_logging(log_file: StoragePath) -> None:
        os.makedirs(os.path.dirname(str(log_file)), exist_ok=True)

        root = logging.getLogger()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s"
        )

        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    h.close()
                except Exception:
                    logging.debug("Ignoring error while closing previous file handler", exc_info=True)
                root.removeHandler(h)

        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        logging.debug(f"Logging setup complete, writing to: {log_file}")
