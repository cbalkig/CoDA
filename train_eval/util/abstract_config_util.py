# abstract_config_util.py
import argparse
import configparser
import sys

from data.file.path import Path, StoragePath


class AbstractConfigUtil:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AbstractConfigUtil, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--cfg_file", type=str, help="Path to configuration file")
        args, _ = parser.parse_known_args()

        if args.cfg_file:
            config_path = StoragePath(Path(args.cfg_file))
        else:
            self.uid = Path(sys.argv[0]).stem
            config_path = StoragePath(Path(f'{self.uid}.cfg'))

        if not config_path.path.exists():
            raise FileNotFoundError(f'Configuration file not found: {config_path}')

        self.cfg = configparser.ConfigParser()
        self.cfg.read(config_path.path)
        self.cfg_file_path = config_path
