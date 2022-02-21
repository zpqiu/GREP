# -*- encoding:utf-8 -*-
"""
Some help functions for logging
"""
import logging


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        logger.addHandler(stream_handler)


def convert_dataset_config(cfg, d):
    d['dataset.name'] = cfg.dataset.name


def convert_model_config(cfg, d):
    d['model.name'] = cfg.model.name
    d['model.dropout'] = cfg.model.dropout
    d['model.neg_count'] = cfg.model.neg_count


def convert_training_config(cfg, d):
    d['train.name'] = cfg.training.name
    d['train.batch_size'] = cfg.training.batch_size
    d['train.accumulate'] = cfg.training.accumulate
    d['train.learning_rate'] = cfg.training.learning_rate
    d['train.epochs'] = cfg.training.epochs
    d['train.seed'] = cfg.training.seed


def convert_omegaconf_to_dict(cfg):
    d = dict()
    convert_dataset_config(cfg, d)
    convert_model_config(cfg, d)
    convert_training_config(cfg, d)

    return d
