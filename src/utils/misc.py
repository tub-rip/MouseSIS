import os
import logging

logger = logging.getLogger(__name__)


def check_key_and_bool(config: dict, key: str) -> bool:
    """Check the existance of the key and if it's True

    Args:
        config (dict): dict.
        key (str): Key name to be checked.

    Returns:
        bool: Return True only if the key exists in the dict and its value is True.
            Otherwise returns False.
    """
    return key in config.keys() and config[key]


def check_file_exists(filename: str) -> bool:
    """Return True if the file exists.

    Args:
        filename (str): _description_

    Returns:
        bool: _description_
    """
    logger.debug(f"Check {filename}")
    res = os.path.exists(filename)
    if not res:
        logger.warning(f"{filename} does not exist!")
    return res


def uniquify_dir(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f'{filename}-{counter}{extension}'
        counter += 1
    return path

COLORS = [
    (33, 150, 243),  # Blue
    (244, 67, 54),   # Red
    (76, 175, 80),   # Green
    (255, 152, 0),   # Orange
    (121, 85, 72),   # Brown
    (158, 158, 158), # Grey
    (96, 125, 139),  # Blue Grey
    (233, 30, 99),   # Pink
    (0, 188, 212),   # Cyan
    (205, 220, 57),  # Lime
    (63, 81, 181),   # Indigo
    (139, 195, 74),  # Light Green
    (255, 193, 7),   # Amber
    (255, 87, 34),   # Deep Orange
    (103, 58, 183)   # Deep Purple
]