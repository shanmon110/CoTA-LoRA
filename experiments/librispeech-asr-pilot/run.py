"""Compatibility launcher. The recipe lives under examples/librispeech/asr."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.librispeech.asr.pilot_3b import main
    main()
