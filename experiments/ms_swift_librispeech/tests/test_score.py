import pytest
import torch

from experiments.ms_swift_librispeech.evaluate import continuations, edit_distance, words


def test_padded_prompt_eos_and_cap():
    prefix = torch.tensor([[0, 5], [4, 6]])
    result = torch.tensor([[0, 5, 10, 8, 0], [4, 6, 7, 9, 10]])
    assert list(continuations(result, prefix, [8], 3)) == [([10], False), ([7, 9, 10], True)]
    with pytest.raises(ValueError, match='prompt prefix'):
        list(continuations(result, prefix.flip(0), [8], 3))
    with pytest.raises(ValueError, match='without EOS'):
        list(continuations(result, prefix, [11], 5))


def test_english_normalization_and_per_utterance_edits():
    assert words(' IT’S two—words. ') == ["it's", 'two', 'words']
    assert edit_distance(words('a b c'), words('A C')) == 1
    assert edit_distance(words('a'), words('')) == 1
