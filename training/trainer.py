
import logging
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LoRATrainer:


    def __init__(self,
                 model,
                 tokenizer,
                 lora_config: Optional[Dict[str, Any]] = None):
        """
        Initialize LoRA trainer.

        Args:
            model: ICHLLM model to train
            tokenizer: Tokenizer for the model
            lora_config: LoRA configuration parameters
        """
        self.model = model
        self.tokenizer = tokenizer

        # Default LoRA configuration
        self.lora_config = {
            "rank": 64,
            "alpha": 16,
            "dropout": 0.1,
            "cultural_scaling": True,
            "num_cultures": 147,
            **lora_config or {}
        }

        # Apply LoRA to model
        self._apply_lora()

        # Training state
        self.training_step = 0
        self.epoch = 0

        logger.info(f"Initialized LoRA trainer with rank={self.lora_config['rank']}")

    def _apply_lora(self):
        """Apply LoRA adaptations to the model."""
        from ..models.lora_layers import LoRAManager

        # Initialize LoRA manager
        self.lora_manager = LoRAManager(
            model=self.model,
            lora_config=self.lora_config
        )

        # Apply LoRA to linear layers
        self.lora_manager.apply_lora_to_model()

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")

    def train(self,
              dataset,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-4,
              cultural_loss_weight: float = 0.1,
              output_dir: str = "./lora_output",
              save_steps: int = 500,
              eval_steps: int = 1000,
              logging_steps: int = 100,
              **kwargs):
        """
        Train the model with LoRA.

        Args:
            dataset: Training dataset (CHITDataset)
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            cultural_loss_weight: Weight for cultural loss component
            output_dir: Output directory for saving model
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
        """
        from .cultural_loss import CulturalLoss
        import os
        from pathlib import Path

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )

        # Initialize cultural loss
        cultural_loss_fn = CulturalLoss(
            cultural_weight=cultural_loss_weight,
            num_cultures=self.lora_config.get("num_cultures", 147)
        )

        # Set model to training mode
        self.model.train()

        # Training loop
        total_steps = len(dataloader) * num_epochs
        global_step = 0

        logger.info(f"Starting training for {num_epochs} epochs ({total_steps} steps)")

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_cultural_loss = 0.0

            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                leave=True
            )

            for step, batch in enumerate(progress_bar):
                # Forward pass
                outputs = self.model(**batch)

                # Calculate standard language modeling loss
                lm_loss = outputs.loss

                # Calculate cultural loss
                cultural_loss = cultural_loss_fn(
                    outputs=outputs,
                    batch=batch,
                    model=self.model
                )

                # Combined loss
                total_loss = lm_loss + cultural_loss_weight * cultural_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Update tracking
                epoch_loss += lm_loss.item()
                epoch_cultural_loss += cultural_loss.item()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "LM Loss": f"{lm_loss.item():.4f}",
                    "Cultural Loss": f"{cultural_loss.item():.4f}",
                    "Total Loss": f"{total_loss.item():.4f}"
                })

                # Logging
                if global_step % logging_steps == 0:
                    logger.info(
                        f"Step {global_step}: LM Loss = {lm_loss.item():.4f}, "
                        f"Cultural Loss = {cultural_loss.item():.4f}, "
                        f"Total Loss = {total_loss.item():.4f}"
                    )

                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_path = output_path / f"checkpoint-{global_step}"
                    self.save_model(checkpoint_path)
                    logger.info(f"Model checkpoint saved to {checkpoint_path}")

                # Evaluation (if eval dataset provided)
                if global_step % eval_steps == 0 and kwargs.get("eval_dataset"):
                    eval_loss = self.evaluate(kwargs["eval_dataset"])
                    logger.info(f"Evaluation loss: {eval_loss:.4f}")
                    self.model.train()  # Return to training mode

            # End of epoch logging
            avg_loss = epoch_loss / len(dataloader)
            avg_cultural_loss = epoch_cultural_loss / len(dataloader)

            logger.info(
                f"Epoch {epoch + 1} completed: "
                f"Avg LM Loss = {avg_loss:.4f}, "
                f"Avg Cultural Loss = {avg_cultural_loss:.4f}"
            )

        # Save final model
        final_path = output_path / "final_model"
        self.save_model(final_path)
        logger.info(f"Training completed. Final model saved to {final_path}")

    def evaluate(self, eval_dataset, batch_size: int = 8) -> float:
        """Evaluate the model on evaluation dataset."""
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def _collate_fn(self, batch):
        """Collate function for DataLoader."""
        # Extract texts and cultural contexts from batch
        texts = []
        cultural_contexts = []

        for sample in batch:
            if hasattr(sample, 'input_text') and hasattr(sample, 'target_text'):
                # For training samples with input/target
                text = f"{sample.input_text} {sample.target_text}"
                texts.append(text)
                cultural_contexts.append(sample.cultural_context)
            elif hasattr(sample, 'instruction') and hasattr(sample, 'output'):
                # For instruction-tuning format
                text = f"Instruction: {sample.instruction}\nResponse: {sample.output}"
                texts.append(text)
                cultural_contexts.append(getattr(sample, 'cultural_context', {}))
            else:
                # Fallback to string representation
                texts.append(str(sample))
                cultural_contexts.append({})

        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move to device if model is on GPU
        if next(self.model.parameters()).is_cuda:
            encodings = {k: v.cuda() for k, v in encodings.items()}

        # Add labels for language modeling
        encodings["labels"] = encodings["input_ids"].clone()

        # Add cultural context information
        encodings["cultural_contexts"] = cultural_contexts

        return encodings

    def save_model(self, save_path: Union[str, Path]):
        """Save the LoRA-adapted model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.lora_manager.save_lora_weights(save_path / "lora_weights.pt")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        # Save training configuration
        config = {
            "lora_config": self.lora_config,
            "training_step": self.training_step,
            "epoch": self.epoch
        }

        import json
        with open(save_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: Union[str, Path]):
        """Load a previously saved LoRA-adapted model."""
        load_path = Path(load_path)

        # Load LoRA weights
        lora_weights_path = load_path / "lora_weights.pt"
        if lora_weights_path.exists():
            self.lora_manager.load_lora_weights(lora_weights_path)
            logger.info(f"LoRA weights loaded from {lora_weights_path}")

        # Load training configuration
        config_path = load_path / "training_config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            self.lora_config.update(config.get("lora_config", {}))
            self.training_step = config.get("training_step", 0)
            self.epoch = config.get("epoch", 0)

            logger.info(f"Training configuration loaded from {config_path}")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_ratio": trainable_params / total_params,
            "lora_rank": self.lora_config["rank"],
            "lora_alpha": self.lora_config["alpha"],
            "cultural_scaling_enabled": self.lora_config.get("cultural_scaling", False),
            "num_cultures": self.lora_config.get("num_cultures", 147),
            "current_epoch": self.epoch,
            "training_step": self.training_step
        }