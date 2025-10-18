
from typing import Any
import torch.nn as nn
import torch.nn.utils.prune as prune


class Prunning:
    """
    Utility class for applying structured pruning to convolutional layers.

    This class supports both static pruning (fixed sparsity) and dynamic pruning
    (gradually increasing sparsity across epochs) using a scheduler. It is intended
    to reduce the number of active parameters in `Conv3d` layers, potentially
    improving inference efficiency at the cost of model capacity.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model whose layers will be pruned.
    config : Any
        Configuration object (typically parsed from YAML/args) that should contain
        pruning-related flags under `config.model`, e.g.:
        
        * `apply_pruning` : bool
            Whether pruning should be applied.
        * `enable_pruning_scheduler` : bool
            If True, sparsity will be increased gradually.
        * `pruning_amount` : float
            Static sparsity ratio to apply when scheduler is disabled.
    initial_sparsity : float, default=0.1
        Starting sparsity ratio when using the pruning scheduler.
    final_sparsity : float, default=0.5
        Final sparsity ratio after `sparsity_epochs`.
    sparsity_epochs : int, default=10
        Number of epochs over which sparsity is increased when scheduler is active.
    pruning_warmup : int, default=2
        Number of warmup epochs before applying pruning.

    Examples
    --------
    >>> model = My3DModel()
    >>> pruner = Prunning(model, config, initial_sparsity=0.2, final_sparsity=0.6)
    >>> for epoch in range(20):
    ...     pruner.update_pruning(epoch)
    """

    def __init__(self, model: nn.Module, config: Any, initial_sparsity: int = 0.1, final_sparsity: float = 0.5, sparsity_epochs: int = 10, pruning_warmup: int = 2) -> None:
        self.model = model
        self.config = config
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.sparsity_epochs = sparsity_epochs
        self.pruning_warmup = pruning_warmup

    def update_pruning(self, epoch: int) -> None:
        """
        Apply pruning to eligible layers for the given epoch.

        If `enable_pruning_scheduler` is set in the config, sparsity will increase
        linearly from `initial_sparsity` to `final_sparsity` across
        `sparsity_epochs`, starting after the warmup period. Otherwise, a fixed
        pruning amount (`pruning_amount`) is applied.

        Parameters
        ----------
        epoch : int
            Current epoch index.

        Notes
        -----
        * Only `nn.Conv3d` layers are pruned.
        * Pruning is applied using L2-norm structured pruning along the output
          channel dimension (`dim=0`).
        """
        if not getattr(self.config.model, "apply_pruning", False):
            return

        # Use scheduler dynamic sparsity
        if getattr(self.config.model, "enable_pruning_scheduler", False):
            pruning_end = self.pruning_warmup + self.sparsity_epochs
            if epoch < self.pruning_warmup or epoch > pruning_end:
                return

            progress = (epoch - self.pruning_warmup) / self.sparsity_epochs
            sparsity = self.initial_sparsity + (
                (self.final_sparsity - self.initial_sparsity) * progress
            )
        else:
            # Static pruning amount
            sparsity = getattr(self.config.model, "pruning_amount", 0.1)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv3d):
                try:
                    prune.ln_structured(
                        module, name="weight", amount=sparsity, n=2, dim=0)
                    print(f"Pruned {name} with sparsity {sparsity:.2f}")
                except Exception as e:
                    print(f"Could not prune {name}: {e}")
