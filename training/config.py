
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 64
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "query", "key", "value", "dense", "fc1", "fc2"
    ])
    cultural_scaling: bool = True
    num_cultures: int = 147


@dataclass
class CulturalLossConfig:
    """Configuration for cultural loss functions."""
    cultural_weight: float = 0.1
    consistency_weight: float = 0.4
    boundary_weight: float = 0.3
    alignment_weight: float = 0.2
    specificity_weight: float = 0.1
    temperature: float = 0.1
    penalty_weight: float = 1.0
    specificity_threshold: float = 0.5


@dataclass
class DataConfig:
    """Configuration for training data."""
    dataset_path: str = "data/chit"
    max_length: int = 512
    batch_size: int = 4
    num_workers: int = 4
    shuffle: bool = True
    task_filter: Optional[List[str]] = None
    culture_filter: Optional[List[str]] = None
    sacred_level_max: int = 2
    train_split: str = "train"
    eval_split: str = "validation"


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "linear"
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999


@dataclass
class TrainingConfig:
    """Main training configuration."""
    # Model configuration
    model_name: str = "ichllm-7b"
    model_path: Optional[str] = None

    # Training parameters
    num_epochs: int = 3
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1

    # Logging and saving
    output_dir: str = "./lora_output"
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 1000
    save_total_limit: int = 3

    # Evaluation
    evaluation_strategy: str = "steps"  # "no", "steps", "epoch"
    eval_dataset_path: Optional[str] = None
    metric_for_best_model: str = "eval_loss"

    # Resource configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 0

    # Reproducibility
    seed: int = 42

    # Component configurations
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    cultural_loss: CulturalLossConfig = field(default_factory=CulturalLossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Cultural training specific
    enable_cultural_loss: bool = True
    enable_sacred_boundary_protection: bool = True
    cultural_validation_steps: int = 1000

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dictionary
        config_dict = self.to_dict()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}

        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                result[key] = value.__dict__.copy()
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # Extract component configurations
        lora_config = LoRAConfig(**config_dict.get('lora', {}))
        cultural_loss_config = CulturalLossConfig(**config_dict.get('cultural_loss', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))

        # Remove component configs from main dict
        main_config = config_dict.copy()
        for key in ['lora', 'cultural_loss', 'data', 'optimization']:
            main_config.pop(key, None)

        # Create main config
        config = cls(
            lora=lora_config,
            cultural_loss=cultural_loss_config,
            data=data_config,
            optimization=optimization_config,
            **main_config
        )

        return config

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []

        # Validate model configuration
        if self.model_name not in ["ichllm-7b", "ichllm-13b"]:
            warnings.append(f"Unknown model name: {self.model_name}")

        # Validate training parameters
        if self.num_epochs <= 0 and (self.max_steps is None or self.max_steps <= 0):
            warnings.append("Either num_epochs or max_steps must be positive")

        if self.lora.rank <= 0:
            warnings.append("LoRA rank must be positive")

        if self.lora.alpha <= 0:
            warnings.append("LoRA alpha must be positive")

        if not (0.0 <= self.lora.dropout <= 1.0):
            warnings.append("LoRA dropout must be between 0 and 1")

        # Validate cultural loss weights
        cultural_weights = [
            self.cultural_loss.consistency_weight,
            self.cultural_loss.boundary_weight,
            self.cultural_loss.alignment_weight,
            self.cultural_loss.specificity_weight
        ]

        if any(w < 0 for w in cultural_weights):
            warnings.append("Cultural loss weights must be non-negative")

        # Validate data configuration
        if self.data.max_length <= 0:
            warnings.append("max_length must be positive")

        if self.data.batch_size <= 0:
            warnings.append("batch_size must be positive")

        if self.data.sacred_level_max < 0 or self.data.sacred_level_max > 3:
            warnings.append("sacred_level_max must be between 0 and 3")

        # Validate optimization configuration
        if self.optimization.learning_rate <= 0:
            warnings.append("learning_rate must be positive")

        if self.optimization.weight_decay < 0:
            warnings.append("weight_decay must be non-negative")

        if self.optimization.max_grad_norm <= 0:
            warnings.append("max_grad_norm must be positive")

        # Validate logging parameters
        if self.logging_steps <= 0:
            warnings.append("logging_steps must be positive")

        if self.save_steps <= 0:
            warnings.append("save_steps must be positive")

        if self.eval_steps <= 0:
            warnings.append("eval_steps must be positive")

        return warnings

    def get_effective_batch_size(self) -> int:
        """Get effective batch size considering gradient accumulation."""
        return self.data.batch_size * self.gradient_accumulation_steps

    def get_training_summary(self) -> str:
        """Get a summary string of training configuration."""
        warnings = self.validate()
        warning_str = f" (Warnings: {len(warnings)})" if warnings else ""

        return f"""Training Configuration Summary{warning_str}:
Model: {self.model_name}
Training: {self.num_epochs} epochs, batch size {self.get_effective_batch_size()}
LoRA: rank={self.lora.rank}, alpha={self.lora.alpha}, dropout={self.lora.dropout}
Cultural Loss: enabled={self.enable_cultural_loss}, weight={self.cultural_loss.cultural_weight}
Learning Rate: {self.optimization.learning_rate}
Output: {self.output_dir}
Sacred Boundary Protection: {self.enable_sacred_boundary_protection}
Supported Cultures: {self.lora.num_cultures}"""


# Predefined configurations for common use cases
def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_quick_test_config() -> TrainingConfig:
    """Get configuration for quick testing."""
    return TrainingConfig(
        num_epochs=1,
        max_steps=100,
        batch_size=2,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        lora=LoRAConfig(rank=8, alpha=8),  # Smaller for faster training
        cultural_loss=CulturalLossConfig(cultural_weight=0.05)  # Lower weight for testing
    )


def get_production_config() -> TrainingConfig:
    """Get configuration for production training."""
    return TrainingConfig(
        num_epochs=5,
        batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=50,
        save_steps=200,
        eval_steps=500,
        lora=LoRAConfig(rank=64, alpha=16),
        cultural_loss=CulturalLossConfig(cultural_weight=0.15),
        optimization=OptimizationConfig(
            learning_rate=3e-4,
            warmup_steps=500,
            weight_decay=0.01
        ),
        fp16=True,  # Enable mixed precision for faster training
        save_total_limit=5
    )


def get_cultural_focus_config() -> TrainingConfig:
    """Get configuration with emphasis on cultural learning."""
    return TrainingConfig(
        num_epochs=4,
        batch_size=6,
        cultural_loss=CulturalLossConfig(
            cultural_weight=0.2,  # Higher cultural loss weight
            consistency_weight=0.5,
            boundary_weight=0.4,
            alignment_weight=0.3,
            specificity_weight=0.2
        ),
        enable_cultural_loss=True,
        enable_sacred_boundary_protection=True,
        cultural_validation_steps=500,  # More frequent cultural validation
        data=DataConfig(sacred_level_max=1)  # More restrictive sacred content filtering
    )
