"""Compatibility entry point for the original single-device experiment driver."""
from pathlib import Path
import runpy

from .local.prepare_data import ACCUM, BATCH, COUNTS, DEFAULT_CONFIG, prepare_splits, read_rows


def main():
    driver = Path(__file__).resolve().parents[3] / "experiments/librispeech-asr-pilot/run_7b.py"
    runpy.run_path(str(driver), run_name="__main__")


if __name__ == "__main__":
    main()
