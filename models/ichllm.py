
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    LlamaModel, LlamaConfig, LlamaTokenizer,
    LlamaForCausalLM, GenerationConfig
)
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging

from .cultural_adapters import CulturalAdapter, SacredBoundaryFilter
from .lora_layers import LoRAAdapter
from .tokenizers import CulturalTokenizer

logger = logging.getLogger(__name__)

@dataclass
class ICHLLMConfig:
    """Configuration for ICHLLM models"""

    # Base model configuration
    base_model: str = "meta-llama/Llama-2-7b-hf"
    model_size: str = "7B"  # "7B" or "13B"

    # Cultural adaptations
    num_cultures: int = 147
    cultural_embedding_dim: int = 512
    enable_cultural_routing: bool = True
    enable_sacred_boundary_protection: bool = True

    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    # Generation configuration
    max_sequence_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # Cultural sensitivity
    cultural_sensitivity_threshold: float = 0.8
    sacred_content_filter_strength: float = 0.9

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class ICHLLM(nn.Module):

    def __init__(self, config: ICHLLMConfig):
        super().__init__()
        self.config = config

        # Load base LLaMA model
        self.base_model = LlamaForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Cultural adaptations
        if config.enable_cultural_routing:
            self.cultural_adapter = CulturalAdapter(
                hidden_size=self.base_model.config.hidden_size,
                num_cultures=config.num_cultures,
                cultural_embedding_dim=config.cultural_embedding_dim
            )
        else:
            self.cultural_adapter = None

        # Sacred boundary protection
        if config.enable_sacred_boundary_protection:
            self.sacred_boundary_filter = SacredBoundaryFilter(
                hidden_size=self.base_model.config.hidden_size,
                filter_strength=config.sacred_content_filter_strength
            )
        else:
            self.sacred_boundary_filter = None

        # LoRA adapters
        self.lora_adapters = nn.ModuleDict()
        self._apply_lora_adapters()

        # Cultural tokenizer
        self.tokenizer = CulturalTokenizer.from_pretrained(
            config.base_model,
            num_cultures=config.num_cultures
        )

        # Generation configuration
        self.generation_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def _apply_lora_adapters(self):
        """Apply LoRA adapters to target modules"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.config.lora_target_modules):
                if isinstance(module, nn.Linear):
                    # Replace linear layer with LoRA adapter
                    lora_adapter = LoRAAdapter(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout,
                        original_layer=module
                    )

                    # Register LoRA adapter
                    self.lora_adapters[name.replace('.', '_')] = lora_adapter

                    # Replace the module
                    parent_name = name.rsplit('.', 1)[0]
                    child_name = name.rsplit('.', 1)[1]
                    parent_module = self.base_model.get_submodule(parent_name)
                    setattr(parent_module, child_name, lora_adapter)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cultural_context: Optional[Dict[str, Any]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )

        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states

        # Apply cultural adaptations
        if self.cultural_adapter and cultural_context:
            hidden_states = self.cultural_adapter(
                hidden_states,
                cultural_context
            )

        # Apply sacred boundary protection
        if self.sacred_boundary_filter:
            hidden_states = self.sacred_boundary_filter(
                hidden_states,
                cultural_context
            )

        # Update outputs with modified hidden states
        # Note: This is a simplified implementation
        # In practice, you'd need to recompute the language modeling head
        if hasattr(outputs, 'logits'):
            # Recompute logits with modified hidden states
            lm_head = self.base_model.lm_head
            logits = lm_head(hidden_states)
            outputs.logits = logits

        return {
            "logits": outputs.logits,
            "hidden_states": hidden_states,
            "loss": outputs.loss if hasattr(outputs, 'loss') else None,
            "cultural_adaptation_applied": self.cultural_adapter is not None,
            "sacred_boundary_protection_applied": self.sacred_boundary_filter is not None
        }

    def generate_culturally_aware(
        self,
        input_text: str,
        cultural_context: Dict[str, Any],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        sacred_boundary_protection: bool = True,
        **generation_kwargs
    ) -> Dict[str, Any]:
   
        # Tokenize input with cultural context
        inputs = self.tokenizer.encode_with_cultural_context(
            input_text,
            cultural_context
        )

        # Update generation config
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature

        # Generate with cultural awareness
        with torch.no_grad():
            input_ids = inputs["input_ids"].unsqueeze(0)
            attention_mask = inputs["attention_mask"].unsqueeze(0)

            # Custom generation loop with cultural adaptations
            generated_ids = self._generate_with_cultural_context(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cultural_context=cultural_context,
                generation_config=gen_config,
                sacred_boundary_protection=sacred_boundary_protection
            )

        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Post-process for cultural appropriateness
        if sacred_boundary_protection and self.sacred_boundary_filter:
            generated_text = self.sacred_boundary_filter.post_process_text(
                generated_text,
                cultural_context
            )

        return {
            "generated_text": generated_text,
            "input_text": input_text,
            "cultural_context": cultural_context,
            "generation_metadata": {
                "model_size": self.config.model_size,
                "cultural_adaptation_used": self.cultural_adapter is not None,
                "sacred_boundary_protection_used": sacred_boundary_protection,
                "num_tokens_generated": len(generated_ids[0]) - len(input_ids[0])
            }
        }

    def _generate_with_cultural_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cultural_context: Dict[str, Any],
        generation_config: GenerationConfig,
        sacred_boundary_protection: bool = True
    ) -> torch.Tensor:
        """Generate tokens with cultural context awareness"""

        batch_size, seq_len = input_ids.shape
        max_new_tokens = generation_config.max_new_tokens
        temperature = generation_config.temperature

        # Initialize generation
        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()

        for step in range(max_new_tokens):
            # Forward pass with cultural context
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=current_attention_mask,
                    cultural_context=cultural_context
                )

            # Get next token logits
            next_token_logits = outputs["logits"][:, -1, :] / temperature

            # Apply cultural filtering if enabled
            if sacred_boundary_protection and self.sacred_boundary_filter:
                next_token_logits = self.sacred_boundary_filter.filter_token_logits(
                    next_token_logits,
                    cultural_context
                )

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Append next token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            new_attention = torch.ones(batch_size, 1, device=generated_ids.device)
            current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)

        return generated_ids

    def get_cultural_embeddings(self, culture: str) -> torch.Tensor:
        """Get cultural embeddings for a specific culture"""
        if self.cultural_adapter:
            return self.cultural_adapter.get_cultural_embedding(culture)
        else:
            return torch.zeros(self.config.cultural_embedding_dim)

    def enable_lora_training(self):
        """Enable LoRA adapters for training"""
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Enable LoRA adapter parameters
        for adapter in self.lora_adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True

        logger.info(f"Enabled LoRA training with {self.count_trainable_parameters()} trainable parameters")

    def disable_lora_training(self):
        """Disable LoRA adapters (inference mode)"""
        for adapter in self.lora_adapters.values():
            for param in adapter.parameters():
                param.requires_grad = False

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_lora_adapters(self, save_path: str):
        """Save only LoRA adapter weights"""
        lora_state_dict = {}
        for name, adapter in self.lora_adapters.items():
            lora_state_dict[name] = adapter.state_dict()

        torch.save({
            "lora_adapters": lora_state_dict,
            "config": self.config,
            "model_info": {
                "base_model": self.config.base_model,
                "model_size": self.config.model_size,
                "num_cultures": self.config.num_cultures
            }
        }, save_path)

        logger.info(f"Saved LoRA adapters to {save_path}")

    def load_lora_adapters(self, load_path: str):
        """Load LoRA adapter weights"""
        checkpoint = torch.load(load_path, map_location="cpu")
        lora_state_dict = checkpoint["lora_adapters"]

        for name, adapter in self.lora_adapters.items():
            if name in lora_state_dict:
                adapter.load_state_dict(lora_state_dict[name])

        logger.info(f"Loaded LoRA adapters from {load_path}")


class ICHLLM7B(ICHLLM):
    """ICHLLM-7B model implementation"""

    def __init__(self, config: Optional[ICHLLMConfig] = None):
        if config is None:
            config = ICHLLMConfig(
                base_model="meta-llama/Llama-2-7b-hf",
                model_size="7B"
            )
        config.model_size = "7B"
        super().__init__(config)


class ICHLLM13B(ICHLLM):
    """ICHLLM-13B model implementation"""

    def __init__(self, config: Optional[ICHLLMConfig] = None):
        if config is None:
            config = ICHLLMConfig(
                base_model="meta-llama/Llama-2-13b-hf",
                model_size="13B"
            )
        config.model_size = "13B"
        super().__init__(config)


class ICHLLMForSequenceClassification(ICHLLM):
    """ICHLLM model for sequence classification tasks"""

    def __init__(self, config: ICHLLMConfig, num_labels: int):
        super().__init__(config)
        self.num_labels = num_labels

        # Classification head
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size,
            num_labels
        )

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cultural_context: Optional[Dict[str, Any]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for classification"""

        # Get base outputs without computing language modeling loss
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cultural_context=cultural_context,
            labels=None,  # Don't compute LM loss
            **kwargs
        )

        # Get last hidden state for classification
        hidden_states = base_outputs["hidden_states"]

        # Use [CLS] token or last token for classification
        if attention_mask is not None:
            # Use last non-padded token
            batch_size = hidden_states.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            pooled_output = hidden_states[range(batch_size), sequence_lengths]
        else:
            # Use last token
            pooled_output = hidden_states[:, -1]

        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Compute classification loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "loss": loss,
            "cultural_adaptation_applied": base_outputs["cultural_adaptation_applied"],
            "sacred_boundary_protection_applied": base_outputs["sacred_boundary_protection_applied"]
        }


class ICHLLMForTokenClassification(ICHLLM):
    """ICHLLM model for token classification (NER) tasks"""

    def __init__(self, config: ICHLLMConfig, num_labels: int):
        super().__init__(config)
        self.num_labels = num_labels

        # Token classification head
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size,
            num_labels
        )

        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cultural_context: Optional[Dict[str, Any]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for token classification"""

        # Get base outputs
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cultural_context=cultural_context,
            labels=None,
            **kwargs
        )

        # Get hidden states for all tokens
        hidden_states = base_outputs["hidden_states"]

        # Apply dropout and classify each token
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        # Compute token classification loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on non-padded tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "loss": loss,
            "cultural_adaptation_applied": base_outputs["cultural_adaptation_applied"],
            "sacred_boundary_protection_applied": base_outputs["sacred_boundary_protection_applied"]
        }


# Factory function
def create_ichllm_model(
    model_size: str = "7B",
    task_type: str = "causal_lm",
    config: Optional[ICHLLMConfig] = None,
    num_labels: Optional[int] = None
) -> ICHLLM:


    if config is None:
        base_model = f"meta-llama/Llama-2-{model_size.lower()}-hf"
        config = ICHLLMConfig(base_model=base_model, model_size=model_size)

    if task_type == "causal_lm":
        if model_size == "7B":
            return ICHLLM7B(config)
        elif model_size == "13B":
            return ICHLLM13B(config)
        else:
            raise ValueError(f"Unsupported model size: {model_size}")

    elif task_type == "sequence_classification":
        if num_labels is None:
            raise ValueError("num_labels must be provided for sequence classification")
        return ICHLLMForSequenceClassification(config, num_labels)

    elif task_type == "token_classification":
        if num_labels is None:
            raise ValueError("num_labels must be provided for token classification")
        return ICHLLMForTokenClassification(config, num_labels)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")


# Example usage
if __name__ == "__main__":
    # Create ICHLLM-7B model
    config = ICHLLMConfig(
        base_model="meta-llama/Llama-2-7b-hf",
        num_cultures=147,
        enable_cultural_routing=True,
        enable_sacred_boundary_protection=True
    )

    model = ICHLLM7B(config)

    # Example cultural generation
    cultural_context = {
        "culture": "japanese",
        "language": "en",
        "cultural_domain": "traditional_arts",
        "sacred_boundary_respect": True
    }

    # This would work with proper model weights and tokenizer
    logger.info(f"ICHLLM-7B model created with {model.count_trainable_parameters()} parameters")
    logger.info("Model supports cultural awareness and sacred boundary protection")