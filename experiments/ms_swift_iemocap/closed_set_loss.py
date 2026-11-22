"""Optional next-letter conditional cross entropy for the IEMOCAP single-turn SFT recipe."""
import torch
import torch.nn.functional as F


def make_closed_set_loss(trainer):
    """Return a Swift compute_loss_func for the current single-device, unpacked experiment."""
    args, template = trainer.args, trainer.template
    if (args.world_size != 1 or args.gradient_accumulation_steps != 1 or template.sequence_parallel_size != 1
            or template.packing or template.padding_free or args.use_logits_to_keep is not False):
        raise ValueError('Closed-set training requires one process, accumulation 1, unpacked batches and full logits')
    symbols = (*'ABCD', '<|im_end|>', '\n')
    encoded = [template.tokenizer.encode(s, add_special_tokens=False) for s in symbols]
    if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != len(symbols):
        raise ValueError('Expected distinct single-token choices, end marker and newline')
    choices = tuple(ids[0] for ids in encoded[:4])
    end, newline = encoded[4][0], encoded[5][0]

    def loss_func(outputs, labels, *, num_items_in_batch=None, loss_scale=None, trainer=None):
        # Swift counts supervised tokens. This objective has one decision per utterance.
        logits = outputs.logits
        if (loss_scale is not None or labels.ndim != 2 or logits.ndim != 3 or not labels.numel()
                or logits.shape[:2] != labels.shape):
            raise ValueError('Expected full [batch, sequence, vocabulary] logits and unweighted labels')
        labels = labels.to(logits.device)
        mask = labels.ne(-100)
        first = mask.long().argmax(dim=1)
        if not ((mask.sum(dim=1) == 3) & (first > 0)).all():
            raise ValueError('Expected a nonempty prompt followed by exactly three supervised tokens')
        row = torch.arange(labels.shape[0], device=logits.device)
        ids = labels.new_tensor(choices)
        matches = labels[row, first, None].eq(ids)
        if not (matches.any(dim=1) & labels[row, first + 1].eq(end) & labels[row, first + 2].eq(newline)).all():
            raise ValueError('Expected consecutive supervised tokens for the choice, end marker and newline')
        scores = logits[row[:, None], (first - 1)[:, None], ids].float()
        return F.cross_entropy(scores, matches.long().argmax(dim=1))

    return loss_func
