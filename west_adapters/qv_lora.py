"""Q/V LoRA with optional rank matching against the TA trainable budget."""
import json

from .base import AdapterMethod, targets


def qv_budget(model, last_n, ta_rank):
    if type(ta_rank) is not int or ta_rank < 1:
        raise ValueError("reference TA rank must be positive")
    budget = ta_rank * sum(p.in_features + p.out_features + 1
                           for p in targets(model, last_n, "o").values())
    per_rank = sum(p.in_features + p.out_features for p in targets(model, last_n, "qv").values())
    low = max(1, budget // per_rank)
    rank = min((low, low + 1), key=lambda r: abs(r * per_rank - budget))
    return rank, budget, rank * per_rank


class QueryValueLoRA(AdapterMethod):
    name = "qv-lora"
    mode = "lora"
    target_set = "qv"
    options = {"match_ta_budget": False}

    def resolve_rank(self, model, options):
        if not options["match_ta_budget"]:
            return options["rank"]
        rank, budget, actual = qv_budget(model, options["last_n"], options["rank"])
        print(json.dumps(dict(reference_ta_rank=options["rank"], ta_parameters=budget, qv_rank=rank,
                              qv_parameters=actual, difference=actual - budget)), flush=True)
        return rank

    def validate_options(self, options):
        if type(options.get("match_ta_budget", False)) is not bool:
            raise ValueError("match_ta_budget must be boolean")

    def checkpoint_options(self, options):
        # Checkpoints store the resolved rank, never recompute the budget at load.
        return {}
