"""EVA rank allocation, calibration subset and nominal-scale restoration."""
import math

import torch
from torch.utils.data import DataLoader, Subset

from .base import AdapterMethod


class EVA(AdapterMethod):
    name = mode = "eva"
    uses_directions = True
    options = {"eva_samples": 128, "eva_rho": 2., "eva_seed": 42, "eva_token_mask": "labels"}

    def validate_options(self, config):
        rho = config.get("eva_rho")
        if (type(rho) not in (int, float) or not math.isfinite(rho) or rho < 1
                or type(config.get("eva_samples")) is not int or config["eva_samples"] < 1
                or type(config.get("eva_seed")) is not int
                or config.get("eva_token_mask", "labels") not in ("labels", "nonpad")):
            raise ValueError("invalid EVA calibration settings")

    def validate_ranks(self, ranks, config):
        if (sum(ranks.values()) != config["rank"] * len(ranks) or
                max(ranks.values()) > round(config["eva_rho"] * config["rank"])):
            raise ValueError("EVA layer_ranks exceed the per-layer cap or violate the global budget")

    def build(self, proj, rank, config):
        if rank > proj.in_features:
            raise ValueError("too many EVA directions for projection")
        # Redistribute rank while preserving the nominal LoRA scale.
        return self.branch(proj, rank, config["alpha"] * (rank / config["rank"]), config)

    def calibrate(self, model, config, data, collate, batch_size, device):
        count = min(len(data), config["eva_samples"])
        seed = config["eva_seed"]
        indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))[:count].tolist()
        subset = Subset(data, indices)
        loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed))
        directions = self.calibrate_batches(model, config, loader, device)
        config["eva_samples"] = count
        return directions

    def calibrate_batches(self, model, config, batches, device, *, forward_model=None):
        from .eva_calibration import calibrate_eva

        return calibrate_eva(model, batches, last_n=config["last_n"], rank=config["rank"],
                             rho=config["eva_rho"], device=device, forward_model=forward_model,
                             use_label_mask=config["eva_token_mask"] == "labels")
