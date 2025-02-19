import yaml
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("config.yml") as f:
    config = yaml.safe_load(f)

def validate_file(file_path: str) -> bool:
    _, ext = os.path.splitext(file_path)
    return ext.lower() in SUPPORTED_EXTENSIONS