import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .models import ICHLLM7B, ICHLLM13B
from .data import CHITDataset
from .evaluation import ICHEBBenchmark, ICHEBEvaluator, CulturalMetricsCalculator
from .training import LoRATrainer

logger = logging.getLogger(__name__)


class MemoriaFramework:


    def __init__(self,
                 model_name: str = "ichllm-7b",
                 model_path: Optional[str] = None,
                 data_path: Optional[str] = None,
                 benchmark_path: Optional[str] = None,
                 device: str = "auto"):
  
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.benchmark_path = benchmark_path
        self.device = device

        # Initialize components
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.benchmark = None
        self.evaluator = None
        self.trainer = None

        # Cultural knowledge and metrics
        self.cultural_metrics = CulturalMetricsCalculator()

        logger.info(f"Initialized MEMORIA framework with {model_name}")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load ICHLLM model."""
        path = model_path or self.model_path

        if self.model_name == "ichllm-7b":
            self.model = ICHLLM7B.from_pretrained(path) if path else ICHLLM7B()
        elif self.model_name == "ichllm-13b":
            self.model = ICHLLM13B.from_pretrained(path) if path else ICHLLM13B()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        # Move to device
        if self.device != "auto":
            self.model = self.model.to(self.device)

        logger.info(f"Loaded {self.model_name} model")

    def load_dataset(self, data_path: Optional[str] = None, split: str = "train") -> None:
        """Load CHIT dataset."""
        path = data_path or self.data_path

        if not path:
            logger.warning("No data path provided. Using sample dataset.")
            path = "sample_data"

        self.dataset = CHITDataset(path, split=split)
        logger.info(f"Loaded CHIT dataset with {len(self.dataset)} samples")

    def load_benchmark(self, benchmark_path: Optional[str] = None) -> None:
        """Load ICHEB benchmark."""
        path = benchmark_path or self.benchmark_path

        if not path:
            logger.warning("No benchmark path provided. Using sample benchmark.")
            path = "sample_benchmark"

        self.benchmark = ICHEBBenchmark(path)
        self.evaluator = ICHEBEvaluator(self.benchmark, self.cultural_metrics)
        logger.info(f"Loaded ICHEB benchmark with {len(self.benchmark.samples)} samples")

    def generate_culturally_aware(self,
                                input_text: str,
                                culture: str,
                                domain: Optional[str] = None,
                                sacred_boundary_protection: bool = True,
                                max_new_tokens: int = 256,
                                temperature: float = 0.7,
                                **kwargs) -> Dict[str, Any]:
   
        if self.model is None:
            self.load_model()

        cultural_context = {
            "culture": culture,
            "domain": domain,
            "tradition_type": "traditional",
            "geographic_region": culture,
            "time_period": "contemporary"
        }

        result = self.model.generate_culturally_aware(
            input_text=input_text,
            cultural_context=cultural_context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            sacred_boundary_protection=sacred_boundary_protection,
            **kwargs
        )

        # Calculate cultural score for the generated text
        cultural_score = self.cultural_metrics.calculate_cultural_score(
            text=result["generated_text"],
            culture=culture,
            expected_cultural_elements=cultural_context,
            domain=domain or "other"
        )

        result["cultural_analysis"] = {
            "cultural_score": cultural_score.overall_score,
            "structural_authenticity": cultural_score.structural_authenticity,
            "motif_fidelity": cultural_score.motif_fidelity,
            "linguistic_authenticity": cultural_score.linguistic_authenticity,
            "value_alignment": cultural_score.value_alignment,
            "transmission_appropriateness": cultural_score.transmission_appropriateness
        }

        return result

    def evaluate_model(self,
                      model: Optional[Any] = None,
                      benchmark: Optional[str] = None,
                      sacred_level_filter: int = 2,
                      batch_size: int = 8,
                      output_path: Optional[str] = None) -> Dict[str, Any]:
    
        if self.evaluator is None:
            self.load_benchmark()

        eval_model = model or self.model
        if eval_model is None:
            self.load_model()
            eval_model = self.model

        # Get benchmark samples
        samples = None
        if benchmark:
            samples = self.benchmark.get_samples(task_filter=[benchmark])

        results = self.evaluator.evaluate_model(
            model=eval_model,
            samples=samples,
            batch_size=batch_size,
            sacred_level_filter=sacred_level_filter
        )

        if output_path:
            self.evaluator.generate_report(results, output_path)
            logger.info(f"Evaluation report saved to {output_path}")

        return results

    def train_lora(self,
                   dataset: Optional[CHITDataset] = None,
                   num_epochs: int = 3,
                   batch_size: int = 4,
                   learning_rate: float = 5e-4,
                   lora_rank: int = 64,
                   lora_alpha: int = 16,
                   cultural_loss_weight: float = 0.1,
                   output_dir: str = "./lora_output",
                   **kwargs) -> None:
     
        if self.model is None:
            self.load_model()

        train_dataset = dataset or self.dataset
        if train_dataset is None:
            self.load_dataset()
            train_dataset = self.dataset

        # Initialize trainer
        from .training import LoRATrainer
        self.trainer = LoRATrainer(
            model=self.model,
            tokenizer=self.model.tokenizer,
            lora_config={
                "rank": lora_rank,
                "alpha": lora_alpha,
                "dropout": 0.1,
                "cultural_scaling": True,
                "num_cultures": 147
            }
        )

        # Train model
        self.trainer.train(
            dataset=train_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            cultural_loss_weight=cultural_loss_weight,
            output_dir=output_dir,
            **kwargs
        )

        logger.info(f"LoRA training completed. Model saved to {output_dir}")

    def get_cultural_statistics(self) -> Dict[str, Any]:
        """Get cultural coverage statistics across all components."""
        stats = {
            "framework_version": "1.0.0",
            "supported_cultures": 147,
            "unesco_domains": 6,
            "task_categories": 6
        }

        if self.dataset:
            dataset_stats = self.dataset.get_statistics()
            stats["dataset"] = {
                "total_samples": dataset_stats["total_samples"],
                "cultures_represented": len(dataset_stats["culture_distribution"]),
                "tasks_covered": len(dataset_stats["task_distribution"]),
                "domains_covered": len(dataset_stats["domain_distribution"])
            }

        if self.benchmark:
            benchmark_stats = self.benchmark.get_statistics()
            stats["benchmark"] = {
                "evaluation_samples": benchmark_stats["total_samples"],
                "cultures_evaluated": benchmark_stats["num_cultures"],
                "tasks_evaluated": benchmark_stats["num_tasks"],
                "domains_evaluated": benchmark_stats["num_domains"]
            }

        if self.model:
            stats["model"] = {
                "model_name": self.model_name,
                "parameters": "7B" if "7b" in self.model_name.lower() else "13B",
                "cultural_adapters": 147,
                "lora_enabled": hasattr(self.model, 'lora_adapters')
            }

        return stats

    def validate_cultural_content(self,
                                text: str,
                                culture: str,
                                domain: str = "other",
                                strict_mode: bool = False) -> Dict[str, Any]:
  
        # Calculate cultural metrics
        cultural_context = {
            "culture": culture,
            "domain": domain,
            "tradition_type": "traditional"
        }

        cultural_score = self.cultural_metrics.calculate_cultural_score(
            text=text,
            culture=culture,
            expected_cultural_elements=cultural_context,
            domain=domain
        )

        # Check for sacred boundary violations
        contains_sacred, violations = self.cultural_metrics.kb.contains_sacred_content(text, culture)

        # Generate validation report
        validation_result = {
            "overall_appropriateness": cultural_score.overall_score,
            "structural_authenticity": cultural_score.structural_authenticity,
            "value_alignment": cultural_score.value_alignment,
            "sacred_boundary_violations": violations,
            "contains_sacred_content": contains_sacred,
            "validation_status": "approved" if cultural_score.overall_score > 0.7 and not contains_sacred else "needs_review",
            "recommendations": []
        }

        # Add recommendations
        if cultural_score.structural_authenticity < 0.6:
            validation_result["recommendations"].append("Review narrative structure and cultural authenticity")

        if cultural_score.value_alignment < 0.7:
            validation_result["recommendations"].append("Improve value alignment with cultural frameworks")

        if contains_sacred_content:
            validation_result["recommendations"].append("Remove or appropriately handle sacred/sensitive content")

        if cultural_score.linguistic_authenticity < 0.5:
            validation_result["recommendations"].append("Add more culturally authentic linguistic features")

        return validation_result

    def __str__(self) -> str:
        """String representation of the framework."""
        return f"MEMORIA Framework v1.0.0 ({self.model_name})"

    def __repr__(self) -> str:
        """Detailed representation of the framework."""
        return (f"MemoriaFramework(model_name='{self.model_name}', "
                f"model_loaded={self.model is not None}, "
                f"dataset_loaded={self.dataset is not None}, "
                f"benchmark_loaded={self.benchmark is not None})")