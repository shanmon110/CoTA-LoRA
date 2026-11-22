"""Length budgets shared by offline manifest audits and CPU collation."""
import math
from dataclasses import dataclass


def add_budget_args(parser):
    parser.add_argument("--max-audio-seconds", type=float, help="Maximum source duration per utterance.")
    parser.add_argument("--max-batch-frames", type=int, help="Maximum sum of valid mel frames in a microbatch.")
    parser.add_argument("--max-sequence-length", type=int, help="Maximum expanded input tokens per utterance.")


@dataclass(frozen=True)
class Budget:
    max_audio_seconds: float = None
    max_batch_frames: int = None
    max_sequence_length: int = None

    def __post_init__(self):
        if self.max_audio_seconds is not None and (
                not math.isfinite(self.max_audio_seconds) or self.max_audio_seconds <= 0):
            raise ValueError("max_audio_seconds must be finite and positive")
        for name in ("max_batch_frames", "max_sequence_length"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_args(cls, args):
        return cls(args.max_audio_seconds, args.max_batch_frames, args.max_sequence_length)

    def reasons(self, *, seconds=None, frames=None, tokens=None):
        observed = (("max_audio_seconds", seconds), ("max_batch_frames", frames),
                    ("max_sequence_length", tokens))
        return [key for key, value in observed if value is not None and
                getattr(self, key) is not None and value > getattr(self, key)]

    def check(self, ids, **lengths):
        reasons = self.reasons(**lengths)
        if reasons:
            raise ValueError(f"{ids}: exceeded {', '.join(reasons)}; observed {lengths}; limits {self}")
