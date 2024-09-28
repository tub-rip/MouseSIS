import argparse
import logging
import os
import sys
import shutil
import subprocess
import uuid
import socket

import yaml

from .misc import uniquify_dir

logger = logging.getLogger(__name__)

LOG_LEVEL = {'debug': logging.DEBUG,
             'info': logging.INFO,
             'warning': logging.WARNING,
             'error': logging.ERROR,
             'critical': logging.CRITICAL}


def parse_args(root, default_path=""):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=default_path,
        help="Config file yaml path", type=str,
    )
    parser.add_argument(
        "--log", help="Log level: [debug, info, warning, error, critical]",
        type=str, default="info"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(root, config["output_dir"])
    output_dir = uniquify_dir(output_dir)
    config["output_dir"] = output_dir
    os.makedirs(config["output_dir"])

    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logger.setLevel(LOG_LEVEL[args.log])
    logging.basicConfig(format='%(asctime)s [%(filename)s] %(levelname)s: %(message)s',
                        level=LOG_LEVEL[args.log],
                        handlers=[
                            logging.FileHandler(f"{config['output_dir']}/main.log", mode="w"),
                            logging.StreamHandler(sys.stdout),
                        ])

    return config, args


def get_config(config_path, root):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(root, config["output_dir"])
    output_dir = uniquify_dir(output_dir)
    config["output_dir"] = output_dir
    os.makedirs(config["output_dir"])

    save_config(output_dir, config_path)

    return config


def save_config(save_dir: str, config_file: str):
    # Copy config file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dest_file = os.path.join(save_dir, "config.yaml")
    shutil.copy(config_file, dest_file)

    # Save runtime information
    runtime_config = fetch_runtime_information()
    with open(os.path.join(save_dir, "runtime.yaml"), "w") as stream:
        # stream = open(os.path.join(save_dir, "runtime.yaml"), "w")
        yaml.dump(runtime_config, stream)


def fetch_runtime_information() -> dict:
    return {"commit": fetch_commit_id(), "server": get_server_name()}


def fetch_commit_id() -> str:
    try:
        label = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return label.decode("utf-8")
    except:
        return "unknown"


def get_server_name() -> str:
    """Get server name based on MAC address. It is hard-coded.
    """
    mac_address = hex(uuid.getnode())
    host_name = socket.gethostname()
    if mac_address == '0x9bbd955b4478' or host_name == 'scioi244-34':
        return "fh_thinkpad"
    return "unknown"
