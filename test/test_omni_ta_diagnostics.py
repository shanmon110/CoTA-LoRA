"""ASR stopping and optional smoke diagnostics with no GPU allocation."""
import json

import pytest
import torch

from examples.omni_talora.evaluate import count_capped_continuations
from examples.omni_talora.diagnostics import SmokeLog


def test_count_only_full_continuations_without_eos():
    # Left-padded prompts may contain EOS. Early-finished rows are right-padded.
    generated = torch.tensor([[0, 2, 4, 8, 9, 9], [0, 2, 4, 2, 0, 0], [2, 3, 4, 8, 9, 2]])
    assert count_capped_continuations(generated, 3, 3, 2) == 1
    assert count_capped_continuations(generated[:, :-1], 3, 3, 2) == 0


def test_smoke_log_tracks_oom_batch_and_resets_peaks(tmp_path, monkeypatch):
    import soundfile as sf
    import numpy as np
    clip = tmp_path / 'short.wav'
    sf.write(clip, np.zeros(1600), 16000)
    path = tmp_path / 'smoke.jsonl'
    resets = []
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', resets.append)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda device: 64)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 128)
    log = SmokeLog(path, 'cuda:1')
    with log.stage('calibration'):
        pass
    with pytest.raises(torch.OutOfMemoryError):
        with log.stage('train'):
            log.collate(lambda rows: {'attention_mask': torch.tensor([[1, 1, 0]])},
                        [{'id': 'utt', 'audio': str(clip)}])
            raise torch.OutOfMemoryError('test')
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(resets) == 2
    assert events[-1]['status'] == 'oom'
    assert events[-1]['stage'] == 'train'
    assert events[-1]['peak_allocated_bytes'] == 64
    assert events[-1]['peak_reserved_bytes'] == 128
    assert events[-1]['samples'] == [dict(id='utt', audio=str(clip), input_tokens=2, input_seconds=0.1)]
