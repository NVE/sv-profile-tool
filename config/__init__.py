import yaml
import os
import logging
from datetime import datetime

class Config:
    def __init__(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(basedir, 'config.yml')
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key):
        keys = key.split('.')
        val = self.config
        for kk in keys:
            if kk in val:
                val = val[kk]
            else:
                raise KeyError(f'Key {key} not found in config file')
        return val
    
    def keys(self):
        return self.config.keys()


class ConfigSecrets:
    def __init__(self):
        basedir = os.path.dirname(os.path.realpath(__file__))
        self.config_path = os.path.join(basedir, 'secrets.yml')
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key):
        keys = key.split('.')
        val = self.config
        for kk in keys:
            if kk in val:
                val = val[kk]
            else:
                raise KeyError(f'Key {key} not found in config file')
        return val

    def keys(self):
        return self.config.keys()


def setup_logger(name='root', log_file=None):
    log_dir = '../logs' if os.path.basename(os.getcwd())=="notebooks" else 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    if log_file is None:
        log_file = f'{log_dir}/app_{datetime.now().strftime("%Y%m%d")}.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger