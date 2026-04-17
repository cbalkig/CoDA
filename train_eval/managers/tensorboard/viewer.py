import logging
import os
from pathlib import Path
from typing import Optional

import yaml  # Requires: pip install pyyaml
from tensorboard import program
from tensorboard.program import TensorBoardPortInUseError


def launch_tensorboard(log_dir: Path, port_number: int = 6007, refresh_interval: Optional[int] = 6007,
                       host: str = "0.0.0.0"):
    tb = program.TensorBoard()
    argv = [
        None,
        '--logdir', str(log_dir.expanduser().resolve().absolute()),
        '--port', str(port_number),
        '--host', host,
    ]

    if refresh_interval:
        argv.extend(['--reload_interval', str(refresh_interval)])

    tb.configure(argv=argv)

    try:
        url = tb.launch()
    except TensorBoardPortInUseError:
        return launch_tensorboard(log_dir, port_number + 1, refresh_interval)

    print(f"-------------------------------------------------------")
    print(f"TensorBoard is running at: {url}")
    print(f"Looking at logs in: {os.path.abspath(log_dir)}")
    if refresh_interval:
        print(f"Refresh interval set to: {refresh_interval} seconds")
    print(f"-------------------------------------------------------")


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    config_file = 'viewer.yaml'

    try:
        config = load_config(config_file)
        tb_config = config.get('tensorboard', {})

        log_dir = Path(tb_config.get('log_dir'))
        port = tb_config.get('port')
        refresh_interval = tb_config.get('refresh_interval')
        host = tb_config.get('host', '0.0.0.0')

        launch_tensorboard(log_dir, port, refresh_interval, host)
        input("Press Enter to quit TensorBoard...")

    except FileNotFoundError as fnf_error:
        logging.error(fnf_error)
    except yaml.YAMLError as yaml_error:
        logging.error(f"Error parsing YAML file: {yaml_error}")
    except Exception as e:
        logging.warning(f"Error launching TensorBoard: {e}")


if __name__ == "__main__":
    main()
