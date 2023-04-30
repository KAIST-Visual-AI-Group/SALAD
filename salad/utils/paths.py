from pathlib import Path
import os

""" project top dir: salad """
PROJECT_DIR = Path(os.path.realpath(__file__)).parents[2]
SALAD_DIR = PROJECT_DIR / "salad"
SPAGHETTI_DIR = SALAD_DIR / "spaghetti"
DATA_DIR = PROJECT_DIR / "data"

if __name__ == "__main__":
    print(PROJECT_DIR)
