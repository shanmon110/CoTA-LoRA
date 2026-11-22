"""Opt-in JSONL profiling for single-device smoke runs."""
import json
import time
from contextlib import contextmanager
from pathlib import Path


class SmokeLog:
    def __init__(self, path, device):
        self.path = Path(path) if path else None
        self.device = str(device)
        self.name = None
        self.samples = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **fields):
        with self.path.open("a", encoding="utf-8") as stream:
            print(json.dumps(dict(stage=self.name, device=self.device, **fields),
                             ensure_ascii=False), file=stream)

    @contextmanager
    def stage(self, name):
        if not self.path:
            yield
            return
        import torch

        self.name = name
        self.samples = []
        cuda = torch.device(self.device).type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.monotonic()
        status = "ok"
        self.write(event="stage_start")
        try:
            yield
            if cuda:
                torch.cuda.synchronize(self.device)
        except torch.OutOfMemoryError:
            status = "oom"
            raise
        except BaseException:
            status = "error"
            raise
        finally:
            self.write(event="stage_end", status=status, seconds=time.monotonic() - start,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if cuda else None,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if cuda else None,
                       samples=self.samples if status != "ok" else [])

    def collate(self, fn, rows):
        """Run in the DataLoader's main process and remember the in-flight batch."""
        if not self.path:
            return fn(rows)
        import soundfile as sf

        self.samples = [dict(id=row.get("id"), audio=row["audio"]) for row in rows]
        for sample in self.samples:
            sample["input_seconds"] = sf.info(sample["audio"]).duration
        batch = fn(rows)
        for sample, tokens in zip(self.samples, batch["attention_mask"].sum(-1).tolist()):
            sample["input_tokens"] = tokens
        self.write(event="batch", samples=self.samples)
        return batch
