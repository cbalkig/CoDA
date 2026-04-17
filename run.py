#!/usr/bin/env python3
"""Unified entrypoint for the CoDA project.

Subcommands
-----------
train_eval   Run the train/eval pipeline. Configs live in train_eval/yamls/.
gen_2d       Run the 3D->2D synthetic image generator (Blender). Configs live
             in image_gen_3d_to_2d/cfgs/.
tb           Launch TensorBoard against a run's log directory.

Examples
--------
    python run.py train_eval --cfg_file train_source.yaml
    python run.py gen_2d --cfg hdr_nature.cfg
    python run.py tb                            # uses viewer.yaml defaults
    python run.py tb --logdir ./runs/my_run/tensorboard
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _resolve_cfg(module_dir: Path, subdir: str, cfg_arg: str) -> Path:
    cfg_path = Path(cfg_arg)
    if cfg_path.is_absolute():
        return cfg_path
    candidate = module_dir / subdir / cfg_arg
    if candidate.exists():
        return candidate
    return module_dir / cfg_arg


def _dispatch(module_dir: Path, cfg: Path, cfg_flag: str) -> None:
    if not cfg.exists():
        sys.exit(f"Config not found: {cfg}")
    os.chdir(module_dir)
    sys.path.insert(0, str(module_dir))
    sys.argv = ["main.py", cfg_flag, str(cfg)]
    runpy.run_path(str(module_dir / "main.py"), run_name="__main__")


def cmd_train_eval(args: argparse.Namespace) -> None:
    module_dir = ROOT / "train_eval"
    cfg = _resolve_cfg(module_dir, "yamls", args.cfg_file)
    _dispatch(module_dir, cfg, "--cfg_file")


def cmd_gen_2d(args: argparse.Namespace) -> None:
    module_dir = ROOT / "image_gen_3d_to_2d"
    cfg = _resolve_cfg(module_dir, "cfgs", args.cfg)
    _dispatch(module_dir, cfg, "--cfg")


def cmd_tb(args: argparse.Namespace) -> None:
    tb_dir = ROOT / "train_eval" / "managers" / "tensorboard"
    sys.path.insert(0, str(ROOT / "train_eval"))

    from managers.tensorboard.viewer import launch_tensorboard, load_config

    if args.logdir is not None:
        log_dir = Path(args.logdir)
        port = args.port
        refresh_interval = args.refresh_interval
        host = args.host
    else:
        cfg = load_config(str(tb_dir / "viewer.yaml")).get("tensorboard", {})
        log_dir = Path(cfg.get("log_dir"))
        port = args.port if args.port is not None else cfg.get("port", 6007)
        refresh_interval = (
            args.refresh_interval
            if args.refresh_interval is not None
            else cfg.get("refresh_interval", 10)
        )
        host = args.host if args.host != "0.0.0.0" else cfg.get("host", "0.0.0.0")

    launch_tensorboard(log_dir, port, refresh_interval, host)
    input("Press Enter to quit TensorBoard...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run", description="CoDA unified launcher")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_te = sub.add_parser("train_eval", help="Run the train/eval pipeline")
    p_te.add_argument(
        "--cfg_file",
        required=True,
        help="YAML name in train_eval/yamls/ (e.g. main.yaml) or an absolute path",
    )
    p_te.set_defaults(func=cmd_train_eval)

    p_g2 = sub.add_parser("gen_2d", help="Run the 3D->2D synthetic image generator")
    p_g2.add_argument(
        "--cfg",
        required=True,
        help="CFG name in image_gen_3d_to_2d/cfgs/ (e.g. dr.cfg) or an absolute path",
    )
    p_g2.set_defaults(func=cmd_gen_2d)

    p_tb = sub.add_parser("tb", help="Launch TensorBoard against a run's log dir")
    p_tb.add_argument(
        "--logdir",
        default=None,
        help="Override log directory (defaults to viewer.yaml's log_dir).",
    )
    p_tb.add_argument("--port", type=int, default=None)
    p_tb.add_argument("--refresh-interval", dest="refresh_interval", type=int, default=None)
    p_tb.add_argument("--host", default="0.0.0.0")
    p_tb.set_defaults(func=cmd_tb)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
