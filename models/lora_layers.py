
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LoRALayer(nn.Module):


    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: int = 16,
        dropout: float = 0.1,
        merge_weights: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights

        # LoRA parameters
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.scaling = alpha / rank
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # Initialize LoRA weights
            self.reset_parameters()

        self.merged = False

    def reset_parameters(self):
        """Initialize LoRA parameters"""
        if hasattr(self, 'lora_A'):
            # Initialize A with random values (Gaussian)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Initialize B with zeros (important for training stability)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA layer"""
        if self.rank == 0:
            return x

        # LoRA computation: x @ A^T @ B^T * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.scaling

    def merge_weights(self):
        """Merge LoRA weights with base weights (for inference)"""
        if self.rank > 0 and not self.merged:
            self.merged = True
            logger.info(f"Merged LoRA weights for layer with rank {self.rank}")

    def unmerge_weights(self):
        """Unmerge LoRA weights (for training)"""
        if self.rank > 0 and self.merged:
            self.merged = False
            logger.info(f"Unmerged LoRA weights for layer with rank {self.rank}")


class LoRALinear(nn.Module):


    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: int = 16,
        dropout: float = 0.1,
        bias: bool = True,
        original_layer: Optional[nn.Linear] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Base linear layer (frozen)
        if original_layer is not None:
            self.linear = original_layer
            # Freeze original weights
            for param in self.linear.parameters():
                param.requires_grad = False
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            # Freeze newly created layer
            for param in self.linear.parameters():
                param.requires_grad = False

        # LoRA adaptation
        self.lora_layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining linear layer and LoRA adaptation"""
        # Base linear transformation
        base_output = self.linear(x)

        # LoRA adaptation
        if self.rank > 0:
            lora_output = self.lora_layer(x)
            return base_output + lora_output
        else:
            return base_output

    def merge_weights(self):
        """Merge LoRA weights with base weights"""
        if self.rank > 0 and not self.lora_layer.merged:
            # Compute merged weight
            merged_weight = self.linear.weight.data + (
                self.lora_layer.lora_B @ self.lora_layer.lora_A * self.lora_layer.scaling
            )
            self.linear.weight.data = merged_weight
            self.lora_layer.merge_weights()

    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights"""
        if self.rank > 0 and self.lora_layer.merged:
            # Subtract LoRA contribution
            lora_contribution = (
                self.lora_layer.lora_B @ self.lora_layer.lora_A * self.lora_layer.scaling
            )
            self.linear.weight.data -= lora_contribution
            self.lora_layer.unmerge_weights()


class LoRAAdapter(nn.Module):
   

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: int = 16,
        dropout: float = 0.1,
        original_layer: Optional[nn.Linear] = None,
        cultural_scaling: bool = False,
        num_cultures: int = 147
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.cultural_scaling = cultural_scaling

        # Core LoRA linear layer
        self.lora_linear = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            original_layer=original_layer
        )

        # Optional cultural-specific scaling
        if cultural_scaling:
            self.cultural_gates = nn.Parameter(
                torch.ones(num_cultures, out_features)
            )
            self.num_cultures = num_cultures
        else:
            self.cultural_gates = None

    def forward(
        self,
        x: torch.Tensor,
        cultural_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """Forward pass with optional cultural scaling"""

        # Apply LoRA linear transformation
        output = self.lora_linear(x)

        # Apply cultural-specific scaling if enabled
        if self.cultural_scaling and self.cultural_gates is not None and cultural_context:
            culture_id = self._get_culture_id(cultural_context.get("culture", "generic"))
            cultural_gate = self.cultural_gates[culture_id]
            output = output * cultural_gate

        return output

    def _get_culture_id(self, culture: str) -> int:
    
        return hash(culture) % self.num_cultures if hasattr(self, 'num_cultures') else 0

    def enable_lora_training(self):
        """Enable LoRA parameters for training"""
        for param in self.lora_linear.lora_layer.parameters():
            param.requires_grad = True

        if self.cultural_gates is not None:
            self.cultural_gates.requires_grad = True

        logger.info(f"Enabled LoRA training for adapter with rank {self.rank}")

    def disable_lora_training(self):
        """Disable LoRA parameters (inference mode)"""
        for param in self.lora_linear.lora_layer.parameters():
            param.requires_grad = False

        if self.cultural_gates is not None:
            self.cultural_gates.requires_grad = False

    def merge_weights(self):
        """Merge LoRA weights for inference"""
        self.lora_linear.merge_weights()

    def unmerge_weights(self):
        """Unmerge LoRA weights for training"""
        self.lora_linear.unmerge_weights()

    def get_lora_parameters(self):
        """Get LoRA parameters for optimization"""
        params = []
        params.extend(self.lora_linear.lora_layer.parameters())
        if self.cultural_gates is not None:
            params.append(self.cultural_gates)
        return params

    def count_lora_parameters(self) -> int:
        """Count trainable LoRA parameters"""
        count = 0
        for param in self.get_lora_parameters():
            count += param.numel()
        return count


class CulturalLoRAConfig:
    """Configuration for Cultural LoRA adapters"""

    def __init__(
        self,
        rank: int = 64,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[list] = None,
        cultural_scaling: bool = True,
        num_cultures: int = 147,
        cultural_regularization: float = 0.01
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.cultural_scaling = cultural_scaling
        self.num_cultures = num_cultures
        self.cultural_regularization = cultural_regularization


class LoRAManager:


    def __init__(self, config: CulturalLoRAConfig):
        self.config = config
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.training_enabled = False

    def add_lora_adapter(
        self,
        name: str,
        in_features: int,
        out_features: int,
        original_layer: Optional[nn.Linear] = None
    ) -> LoRAAdapter:
        """Add a LoRA adapter for a specific layer"""

        adapter = LoRAAdapter(
            in_features=in_features,
            out_features=out_features,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            original_layer=original_layer,
            cultural_scaling=self.config.cultural_scaling,
            num_cultures=self.config.num_cultures
        )

        self.adapters[name] = adapter
        logger.info(f"Added LoRA adapter '{name}' with {adapter.count_lora_parameters()} parameters")

        return adapter

    def enable_lora_training(self):
        """Enable training for all LoRA adapters"""
        for adapter in self.adapters.values():
            adapter.enable_lora_training()
        self.training_enabled = True
        logger.info(f"Enabled training for {len(self.adapters)} LoRA adapters")

    def disable_lora_training(self):
        """Disable training for all LoRA adapters"""
        for adapter in self.adapters.values():
            adapter.disable_lora_training()
        self.training_enabled = False

    def merge_all_weights(self):
        """Merge all LoRA weights for inference"""
        for adapter in self.adapters.values():
            adapter.merge_weights()
        logger.info("Merged all LoRA weights")

    def unmerge_all_weights(self):
        """Unmerge all LoRA weights for training"""
        for adapter in self.adapters.values():
            adapter.unmerge_weights()
        logger.info("Unmerged all LoRA weights")

    def get_all_lora_parameters(self):
        """Get all LoRA parameters for optimization"""
        params = []
        for adapter in self.adapters.values():
            params.extend(adapter.get_lora_parameters())
        return params

    def count_total_lora_parameters(self) -> int:
        """Count total trainable LoRA parameters"""
        return sum(adapter.count_lora_parameters() for adapter in self.adapters.values())

    def compute_cultural_regularization_loss(self) -> torch.Tensor:
        """Compute cultural regularization loss for LoRA adapters"""
        if not self.config.cultural_scaling:
            return torch.tensor(0.0)

        reg_loss = torch.tensor(0.0)
        count = 0

        for adapter in self.adapters.values():
            if adapter.cultural_gates is not None:
                # Encourage diversity in cultural scaling
                gates = adapter.cultural_gates  # [num_cultures, out_features]

                # L2 regularization on gates
                l2_reg = torch.mean(gates ** 2)
                reg_loss += l2_reg

                # Diversity regularization (encourage different cultures to have different scaling)
                if gates.shape[0] > 1:
                    # Compute pairwise similarities and penalize high similarity
                    normalized_gates = F.normalize(gates, dim=1)
                    similarity_matrix = torch.mm(normalized_gates, normalized_gates.t())
                    # Remove diagonal (self-similarity)
                    mask = torch.eye(similarity_matrix.shape[0], device=gates.device)
                    similarity_matrix = similarity_matrix * (1 - mask)
                    diversity_loss = torch.mean(similarity_matrix ** 2)
                    reg_loss += diversity_loss

                count += 1

        if count > 0:
            reg_loss = reg_loss / count * self.config.cultural_regularization

        return reg_loss

    def save_lora_adapters(self, save_path: str):
        """Save all LoRA adapter states"""
        adapter_states = {}
        for name, adapter in self.adapters.items():
            adapter_states[name] = {
                "lora_A": adapter.lora_linear.lora_layer.lora_A.data,
                "lora_B": adapter.lora_linear.lora_layer.lora_B.data,
                "cultural_gates": adapter.cultural_gates.data if adapter.cultural_gates is not None else None,
                "config": {
                    "rank": adapter.rank,
                    "alpha": adapter.alpha,
                    "in_features": adapter.in_features,
                    "out_features": adapter.out_features,
                    "cultural_scaling": adapter.cultural_scaling
                }
            }

        torch.save({
            "adapter_states": adapter_states,
            "global_config": self.config.__dict__,
            "total_parameters": self.count_total_lora_parameters()
        }, save_path)

        logger.info(f"Saved {len(self.adapters)} LoRA adapters to {save_path}")

    def load_lora_adapters(self, load_path: str):
        """Load LoRA adapter states"""
        checkpoint = torch.load(load_path, map_location="cpu")
        adapter_states = checkpoint["adapter_states"]

        for name, state in adapter_states.items():
            if name in self.adapters:
                adapter = self.adapters[name]
                adapter.lora_linear.lora_layer.lora_A.data = state["lora_A"]
                adapter.lora_linear.lora_layer.lora_B.data = state["lora_B"]

                if state["cultural_gates"] is not None and adapter.cultural_gates is not None:
                    adapter.cultural_gates.data = state["cultural_gates"]

        logger.info(f"Loaded {len(adapter_states)} LoRA adapters from {load_path}")

    def print_adapter_statistics(self):
        """Print statistics about LoRA adapters"""
        total_params = self.count_total_lora_parameters()
        logger.info(f"LoRA Adapter Statistics:")
        logger.info(f"  Number of adapters: {len(self.adapters)}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Training enabled: {self.training_enabled}")
        logger.info(f"  Cultural scaling: {self.config.cultural_scaling}")

        for name, adapter in self.adapters.items():
            logger.info(f"  {name}: {adapter.count_lora_parameters():,} parameters")


# Utility functions for LoRA integration

def replace_linear_with_lora(
    model: nn.Module,
    config: CulturalLoRAConfig,
    target_modules: Optional[list] = None
) -> LoRAManager:
 

    lora_manager = LoRAManager(config)
    target_modules = target_modules or config.target_modules

    # Find and replace target modules
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            # Get parent module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name

            if parent_name:
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module = model

            # Create LoRA adapter
            lora_adapter = lora_manager.add_lora_adapter(
                name=name,
                in_features=module.in_features,
                out_features=module.out_features,
                original_layer=module
            )

            # Replace the module
            setattr(parent_module, child_name, lora_adapter)

    logger.info(f"Replaced {len(lora_manager.adapters)} linear layers with LoRA adapters")
    return lora_manager


def get_lora_optimizer(
    lora_manager: LoRAManager,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
  

    lora_params = lora_manager.get_all_lora_parameters()

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    logger.info(f"Created optimizer for {len(lora_params)} LoRA parameter groups")
    logger.info(f"Total LoRA parameters: {lora_manager.count_total_lora_parameters():,}")

    return optimizer


# Example usage and testing
if __name__ == "__main__":
    # Test LoRA adapter
    config = CulturalLoRAConfig(
        rank=64,
        alpha=16,
        dropout=0.1,
        cultural_scaling=True,
        num_cultures=147
    )

    # Create test linear layer
    test_linear = nn.Linear(768, 768)

    # Create LoRA adapter
    lora_adapter = LoRAAdapter(
        in_features=768,
        out_features=768,
        rank=64,
        alpha=16,
        original_layer=test_linear,
        cultural_scaling=True,
        num_cultures=147
    )

    # Test forward pass
    x = torch.randn(2, 10, 768)
    cultural_context = {"culture": "japanese"}

    with torch.no_grad():
        output = lora_adapter(x, cultural_context)

    logger.info(f"LoRA adapter test successful")
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"LoRA parameters: {lora_adapter.count_lora_parameters():,}")
