
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .cultural_metrics import CulturalMetricsCalculator
from .task_evaluators import (
    HeritageNEREvaluator,
    CultureQAEvaluator,
    StoryGenerationEvaluator,
    TranslationEvaluator,
    ClassificationEvaluator,
    DialogueEvaluator
)

logger = logging.getLogger(__name__)


@dataclass
class ICHEBSample:
    """Single evaluation sample in ICHEB benchmark."""
    id: str
    task: str
    culture: str
    domain: str  # UNESCO ICH domain
    input_text: str
    expected_output: str
    cultural_context: Dict[str, Any]
    sacred_level: int = 0  # 0=public, 1=sensitive, 2=sacred, 3=forbidden
    complexity_level: int = 1  # 1=basic, 2=intermediate, 3=advanced
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ICHEBResult:
    """Results for a single ICHEB evaluation sample."""
    sample_id: str
    task: str
    culture: str
    predicted_output: str
    expected_output: str
    cultural_score: float
    cultural_fidelity: Optional[float] = None
    task_specific_scores: Dict[str, float] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)


class ICHEBBenchmark:
    """
    ICHEB benchmark dataset with evaluation samples across 6 tasks and 147 cultures.

    Tasks:
    1. Heritage-NER: Named entity recognition for cultural heritage
    2. Culture-QA: Question answering about cultural practices
    3. Story-Gen: Culturally authentic story generation
    4. Translation: Cross-cultural text translation
    5. Classification: Cultural domain/tradition classification
    6. Dialogue: Cultural context-aware dialogue
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.samples: List[ICHEBSample] = []
        self.task_distribution = {}
        self.culture_distribution = {}
        self.domain_distribution = {}

        # ICHEB task categories
        self.tasks = [
            "heritage_ner", "culture_qa", "story_generation",
            "translation", "classification", "dialogue"
        ]

        # UNESCO ICH domains
        self.domains = [
            "oral_traditions", "performing_arts", "social_practices",
            "traditional_craftsmanship", "knowledge_practices", "other"
        ]

        # Load benchmark data
        self._load_benchmark_data()

    def _load_benchmark_data(self):
        """Load ICHEB benchmark samples from data files."""
        benchmark_file = self.data_path / "icheb_benchmark.json"

        if not benchmark_file.exists():
            logger.warning(f"ICHEB benchmark file not found at {benchmark_file}")
            self._create_sample_benchmark()
            return

        with open(benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for sample_data in data["samples"]:
            sample = ICHEBSample(**sample_data)
            self.samples.append(sample)

        self._compute_distributions()
        logger.info(f"Loaded {len(self.samples)} ICHEB samples")

    def _create_sample_benchmark(self):
        """Create sample ICHEB benchmark data for demonstration."""
        logger.info("Creating sample ICHEB benchmark data...")

        sample_cultures = ["chinese", "japanese", "tibetan", "maasai", "aboriginal_australian"]
        sample_tasks = ["heritage_ner", "culture_qa", "story_generation"]

        sample_data = []
        for i, culture in enumerate(sample_cultures):
            for j, task in enumerate(sample_tasks):
                sample = ICHEBSample(
                    id=f"icheb_{i}_{j}",
                    task=task,
                    culture=culture,
                    domain="oral_traditions" if j == 0 else "social_practices",
                    input_text=f"Sample input for {culture} {task}",
                    expected_output=f"Expected output for {culture} {task}",
                    cultural_context={
                        "tradition_type": "traditional",
                        "geographic_region": culture,
                        "time_period": "historical"
                    },
                    sacred_level=1,
                    complexity_level=2
                )
                self.samples.append(sample)
                sample_data.append({
                    "id": sample.id,
                    "task": sample.task,
                    "culture": sample.culture,
                    "domain": sample.domain,
                    "input_text": sample.input_text,
                    "expected_output": sample.expected_output,
                    "cultural_context": sample.cultural_context,
                    "sacred_level": sample.sacred_level,
                    "complexity_level": sample.complexity_level,
                    "metadata": sample.metadata
                })

        # Save sample data
        self.data_path.mkdir(parents=True, exist_ok=True)
        with open(self.data_path / "icheb_benchmark.json", 'w', encoding='utf-8') as f:
            json.dump({"samples": sample_data}, f, ensure_ascii=False, indent=2)

        self._compute_distributions()

    def _compute_distributions(self):
        """Compute distribution statistics across tasks, cultures, and domains."""
        self.task_distribution = {}
        self.culture_distribution = {}
        self.domain_distribution = {}

        for sample in self.samples:
            self.task_distribution[sample.task] = self.task_distribution.get(sample.task, 0) + 1
            self.culture_distribution[sample.culture] = self.culture_distribution.get(sample.culture, 0) + 1
            self.domain_distribution[sample.domain] = self.domain_distribution.get(sample.domain, 0) + 1

    def get_samples(self,
                   task_filter: Optional[List[str]] = None,
                   culture_filter: Optional[List[str]] = None,
                   domain_filter: Optional[List[str]] = None,
                   sacred_level_max: Optional[int] = None,
                   complexity_level: Optional[int] = None) -> List[ICHEBSample]:
        """Get filtered benchmark samples."""
        filtered_samples = []

        for sample in self.samples:
            # Apply filters
            if task_filter and sample.task not in task_filter:
                continue
            if culture_filter and sample.culture not in culture_filter:
                continue
            if domain_filter and sample.domain not in domain_filter:
                continue
            if sacred_level_max is not None and sample.sacred_level > sacred_level_max:
                continue
            if complexity_level is not None and sample.complexity_level != complexity_level:
                continue

            filtered_samples.append(sample)

        return filtered_samples

    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        return {
            "total_samples": len(self.samples),
            "num_tasks": len(self.task_distribution),
            "num_cultures": len(self.culture_distribution),
            "num_domains": len(self.domain_distribution),
            "task_distribution": self.task_distribution,
            "culture_distribution": self.culture_distribution,
            "domain_distribution": self.domain_distribution,
            "avg_samples_per_task": np.mean(list(self.task_distribution.values())),
            "avg_samples_per_culture": np.mean(list(self.culture_distribution.values())),
            "sacred_level_distribution": {
                level: sum(1 for s in self.samples if s.sacred_level == level)
                for level in range(4)
            },
            "complexity_distribution": {
                level: sum(1 for s in self.samples if s.complexity_level == level)
                for level in range(1, 4)
            }
        }


class ICHEBEvaluator:
    """
    Main evaluator for ICHEB benchmark using Cultural Score and task-specific metrics.
    """

    def __init__(self,
                 benchmark: ICHEBBenchmark,
                 cultural_metrics_calculator: Optional[CulturalMetricsCalculator] = None):
        self.benchmark = benchmark
        self.cultural_metrics = cultural_metrics_calculator or CulturalMetricsCalculator()

        # Initialize task-specific evaluators
        self.task_evaluators = {
            "heritage_ner": HeritageNEREvaluator(),
            "culture_qa": CultureQAEvaluator(),
            "story_generation": StoryGenerationEvaluator(),
            "translation": TranslationEvaluator(),
            "classification": ClassificationEvaluator(),
            "dialogue": DialogueEvaluator()
        }

    def evaluate_model(self,
                      model,
                      samples: Optional[List[ICHEBSample]] = None,
                      batch_size: int = 8,
                      sacred_level_filter: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a model on ICHEB benchmark.

        Args:
            model: Model to evaluate (should have generate_culturally_aware method)
            samples: Specific samples to evaluate (default: all samples)
            batch_size: Batch size for evaluation
            sacred_level_filter: Maximum sacred level to include (for ethical evaluation)

        Returns:
            Comprehensive evaluation results with Cultural Score and task metrics
        """
        if samples is None:
            samples = self.benchmark.get_samples(sacred_level_max=sacred_level_filter)

        results = []

        logger.info(f"Evaluating {len(samples)} samples across {len(set(s.task for s in samples))} tasks")

        # Process samples in batches
        for i in tqdm(range(0, len(samples), batch_size), desc="Evaluating ICHEB"):
            batch_samples = samples[i:i + batch_size]
            batch_results = self._evaluate_batch(model, batch_samples)
            results.extend(batch_results)

        # Aggregate results
        aggregated_results = self._aggregate_results(results)

        return aggregated_results

    def _evaluate_batch(self, model, samples: List[ICHEBSample]) -> List[ICHEBResult]:
        """Evaluate a batch of samples."""
        batch_results = []

        for sample in samples:
            try:
                # Generate model prediction
                generation_result = model.generate_culturally_aware(
                    input_text=sample.input_text,
                    cultural_context=sample.cultural_context,
                    max_new_tokens=512,
                    temperature=0.7,
                    sacred_boundary_protection=True
                )

                predicted_output = generation_result["generated_text"]

                # Calculate Cultural Score
                cultural_score = self.cultural_metrics.calculate_cultural_score(
                    text=predicted_output,
                    culture=sample.culture,
                    expected_cultural_elements=sample.cultural_context,
                    domain=sample.domain
                )

                # Calculate Cultural Fidelity (for translation tasks)
                cultural_fidelity = None
                if sample.task == "translation" and "source_culture" in sample.cultural_context:
                    cultural_fidelity = self.cultural_metrics.calculate_cultural_fidelity(
                        source_text=sample.input_text,
                        translated_text=predicted_output,
                        source_culture=sample.cultural_context["source_culture"],
                        target_culture=sample.culture
                    )

                # Calculate task-specific scores
                task_evaluator = self.task_evaluators.get(sample.task)
                task_specific_scores = {}

                if task_evaluator:
                    task_specific_scores = task_evaluator.evaluate(
                        prediction=predicted_output,
                        reference=sample.expected_output,
                        cultural_context=sample.cultural_context
                    )

                result = ICHEBResult(
                    sample_id=sample.id,
                    task=sample.task,
                    culture=sample.culture,
                    predicted_output=predicted_output,
                    expected_output=sample.expected_output,
                    cultural_score=cultural_score.overall_score,
                    cultural_fidelity=cultural_fidelity.overall_score if cultural_fidelity else None,
                    task_specific_scores=task_specific_scores,
                    error_analysis=generation_result.get("cultural_warnings", {})
                )

                batch_results.append(result)

            except Exception as e:
                logger.error(f"Error evaluating sample {sample.id}: {str(e)}")
                # Create error result
                error_result = ICHEBResult(
                    sample_id=sample.id,
                    task=sample.task,
                    culture=sample.culture,
                    predicted_output="",
                    expected_output=sample.expected_output,
                    cultural_score=0.0,
                    cultural_fidelity=None,
                    task_specific_scores={},
                    error_analysis={"error": str(e)}
                )
                batch_results.append(error_result)

        return batch_results

    def _aggregate_results(self, results: List[ICHEBResult]) -> Dict[str, Any]:
        """Aggregate evaluation results across tasks and cultures."""
        aggregated = {
            "overall": {},
            "by_task": {},
            "by_culture": {},
            "by_domain": {},
            "detailed_results": results
        }

        # Overall statistics
        cultural_scores = [r.cultural_score for r in results]
        cultural_fidelities = [r.cultural_fidelity for r in results if r.cultural_fidelity is not None]

        aggregated["overall"] = {
            "cultural_score_mean": np.mean(cultural_scores),
            "cultural_score_std": np.std(cultural_scores),
            "cultural_fidelity_mean": np.mean(cultural_fidelities) if cultural_fidelities else None,
            "cultural_fidelity_std": np.std(cultural_fidelities) if cultural_fidelities else None,
            "total_samples": len(results),
            "error_rate": sum(1 for r in results if "error" in r.error_analysis) / len(results)
        }

        # Task-specific aggregation
        for task in set(r.task for r in results):
            task_results = [r for r in results if r.task == task]
            task_cultural_scores = [r.cultural_score for r in task_results]

            aggregated["by_task"][task] = {
                "cultural_score_mean": np.mean(task_cultural_scores),
                "cultural_score_std": np.std(task_cultural_scores),
                "samples": len(task_results)
            }

            # Add task-specific metrics
            if task_results and task_results[0].task_specific_scores:
                for metric_name in task_results[0].task_specific_scores.keys():
                    metric_values = [
                        r.task_specific_scores.get(metric_name, 0.0)
                        for r in task_results
                    ]
                    aggregated["by_task"][task][f"{metric_name}_mean"] = np.mean(metric_values)
                    aggregated["by_task"][task][f"{metric_name}_std"] = np.std(metric_values)

        # Culture-specific aggregation
        for culture in set(r.culture for r in results):
            culture_results = [r for r in results if r.culture == culture]
            culture_cultural_scores = [r.cultural_score for r in culture_results]

            aggregated["by_culture"][culture] = {
                "cultural_score_mean": np.mean(culture_cultural_scores),
                "cultural_score_std": np.std(culture_cultural_scores),
                "samples": len(culture_results),
                "tasks": list(set(r.task for r in culture_results))
            }

        return aggregated

    def generate_report(self, results: Dict[str, Any], output_path: str):
        """Generate detailed evaluation report."""
        report_path = Path(output_path)
        report_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        with open(report_path / "icheb_results.json", 'w', encoding='utf-8') as f:
            # Convert results to serializable format
            serializable_results = {
                "overall": results["overall"],
                "by_task": results["by_task"],
                "by_culture": results["by_culture"],
                "detailed_results": [
                    {
                        "sample_id": r.sample_id,
                        "task": r.task,
                        "culture": r.culture,
                        "cultural_score": r.cultural_score,
                        "cultural_fidelity": r.cultural_fidelity,
                        "task_specific_scores": r.task_specific_scores,
                        "error_analysis": r.error_analysis
                    }
                    for r in results["detailed_results"]
                ]
            }
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)

        # Generate summary report
        with open(report_path / "icheb_summary.txt", 'w', encoding='utf-8') as f:
            f.write("ICHEB Evaluation Results Summary\n")
            f.write("=" * 40 + "\n\n")

            overall = results["overall"]
            f.write(f"Overall Performance:\n")
            f.write(f"  Cultural Score: {overall['cultural_score_mean']:.3f} ± {overall['cultural_score_std']:.3f}\n")
            if overall['cultural_fidelity_mean']:
                f.write(f"  Cultural Fidelity: {overall['cultural_fidelity_mean']:.3f} ± {overall['cultural_fidelity_std']:.3f}\n")
            f.write(f"  Total Samples: {overall['total_samples']}\n")
            f.write(f"  Error Rate: {overall['error_rate']:.2%}\n\n")

            f.write("Performance by Task:\n")
            for task, metrics in results["by_task"].items():
                f.write(f"  {task}:\n")
                f.write(f"    Cultural Score: {metrics['cultural_score_mean']:.3f} ± {metrics['cultural_score_std']:.3f}\n")
                f.write(f"    Samples: {metrics['samples']}\n")

            f.write("\nTop Performing Cultures:\n")
            culture_scores = [(culture, metrics['cultural_score_mean'])
                            for culture, metrics in results["by_culture"].items()]
            culture_scores.sort(key=lambda x: x[1], reverse=True)

            for culture, score in culture_scores[:10]:
                f.write(f"  {culture}: {score:.3f}\n")

        logger.info(f"ICHEB evaluation report saved to {report_path}")
