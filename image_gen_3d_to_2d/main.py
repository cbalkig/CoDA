from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config_loader import ConfigLoader
from render_pipeline import RenderPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=Path,
        required=True,
        help="Path to the config file"
    )
    args = parser.parse_args()

    cfg = ConfigLoader(args.cfg)

    log_format = "%(asctime)s │ %(levelname)s │ %(filename)s │ %(funcName)s | %(message)s"
    date_format = "%H:%M:%S"

    cfg.default_cfg.log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.WARNING,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(cfg.default_cfg.log_file, mode='w'),  # or 'w' to overwrite
            logging.StreamHandler()  # Console
        ]
    )

    pipeline = RenderPipeline(cfg)
    pipeline.run()
