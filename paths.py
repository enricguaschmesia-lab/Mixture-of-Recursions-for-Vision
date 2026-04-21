import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

SAVE_DIR = os.environ.get(
    "MOR_SAVE_DIR", os.path.join(PROJECT_ROOT, "results")
)

HF_CACHE_DIR = os.environ.get(
    "HF_HOME", os.path.join(PROJECT_ROOT, "hf_cache")
)

DATA_DIR = os.environ.get(
    "MOR_DATA_DIR", os.path.join(PROJECT_ROOT, "hf_datasets")
)

MODEL_DIR = os.environ.get(
    "MOR_MODEL_DIR", os.path.join(PROJECT_ROOT, "hf_models")
)
