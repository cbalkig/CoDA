from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm

from config_loader import ConfigLoader
from render_logger import RenderLogger
from renderer import USDZRenderer


class RenderPipeline:
    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg

        # ── folders ─────────────────────────────────────────────────────
        self.src = self.cfg.default_cfg.source_folder
        self.dst = self.cfg.default_cfg.destination_folder
        self.dst.mkdir(parents=True, exist_ok=True)

        # wipe destination folder if requested --------------------------
        if self.cfg.default_cfg.cleanup and self.dst.exists():
            logging.warning("cleanup on – wiping %s", self.dst)
            shutil.rmtree(self.dst)
            self.dst.mkdir()

        dest_path = Path(cfg.default_cfg.destination_folder)
        dest_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(cfg.path, dest_path / cfg.path.name)

        success_csv_path = (
            self.cfg.default_cfg.success_csv_path
            if self.cfg.default_cfg.success_csv_path.is_absolute()
            else self.dst / self.cfg.default_cfg.success_csv_path
        )

        failures_csv_path = (
            self.cfg.default_cfg.failures_csv_path
            if self.cfg.default_cfg.failures_csv_path.is_absolute()
            else self.dst / self.cfg.default_cfg.failures_csv_path
        )

        if self.cfg.default_cfg.cleanup:
            if failures_csv_path.exists():
                logging.info("cleanup on – removing previous %s", failures_csv_path)
                failures_csv_path.unlink()
            if success_csv_path.exists():
                logging.info("cleanup on – removing previous %s", success_csv_path)
                success_csv_path.unlink()

        self.log = RenderLogger(success_csv_path, failures_csv_path)

    # -----------------------------------------------------------------
    @staticmethod
    def _md5(p: Path, chunk: int = 8192) -> str:
        h = hashlib.md5()
        with p.open("rb") as f:
            while blk := f.read(chunk):
                h.update(blk)
        return h.hexdigest()

    def _discover(self) -> List[Path]:
        return sorted(self.src.rglob("*.usdz"))

    # -----------------------------------------------------------------
    def run(self):
        if self.cfg.default_cfg.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        rnd = USDZRenderer(self.cfg)
        expect = self.cfg.render_cfg.samples  # joint-sampling count

        models = self._discover()
        for m in tqdm(models, desc="Rendering models", total=len(models)):
            md5 = self._md5(m)
            if self.log.has_all(md5, expect):
                continue

            category = m.parent.name

            produced = rnd.render_model(m, self.dst, category, md5, self.log)
            logging.info("%s → +%d", m.name, produced)

        self.log.close()
