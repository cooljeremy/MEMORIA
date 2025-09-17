
import json
import torch
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

@dataclass
class CHITSample:
    """Single sample in the CHIT dataset"""
    instruction: str
    input_text: str
    output: str
    task_type: str
    culture: str
    language: str
    cultural_context: Dict[str, Any]
    source_dataset: str
    quality_score: float
    sacred_boundary_flags: List[str]
    metadata: Dict[str, Any]

@dataclass
class CHITStatistics:
    """Statistics for the CHIT dataset"""
    total_samples: int = 158000
    num_cultures: int = 147
    num_datasets: int = 13
    num_tasks: int = 6

    # Geographic distribution (from paper Table)
    geographic_distribution: Dict[str, float] = None

    # Language family distribution
    language_family_distribution: Dict[str, float] = None

    # UNESCO domain distribution
    unesco_domain_distribution: Dict[str, float] = None

    def __post_init__(self):
        if self.geographic_distribution is None:
            self.geographic_distribution = {
                "Asia-Pacific": 0.371,    # 58,640 samples
                "Europe": 0.220,          # 34,760 samples
                "Africa": 0.170,          # 26,860 samples
                "Americas": 0.140,        # 22,140 samples
                "Arab States": 0.099      # 15,600 samples
            }

        if self.language_family_distribution is None:
            self.language_family_distribution = {
                "Indo-European": 0.33,    # 52,140 samples
                "Sino-Tibetan": 0.22,     # 34,760 samples
                "Niger-Congo": 0.13,      # 20,540 samples
                "Austronesian": 0.12,     # 18,960 samples
                "Afro-Asiatic": 0.10,     # 15,800 samples
                "Others": 0.10            # 15,800 samples
            }

        if self.unesco_domain_distribution is None:
            self.unesco_domain_distribution = {
                "Oral traditions": 0.30,      # 47,400 samples
                "Performing arts": 0.23,      # 36,340 samples
                "Social practices": 0.21,     # 33,180 samples
                "Traditional crafts": 0.17,   # 26,860 samples
                "Nature knowledge": 0.09      # 14,220 samples
            }


class CHITDataset(Dataset):
    """
    Cultural Heritage Instruction Tuning Dataset

    Features:
    - 158K instruction-response pairs
    - 13 source datasets
    - 6 task types (Heritage-NER, Culture-QA, Story-Gen, Heritage-Trans, etc.)
    - 147 cultural traditions
    - Cultural sensitivity filtering
    - Sacred boundary protection
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        task_filter: Optional[List[str]] = None,
        culture_filter: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        sacred_boundary_protection: bool = True,
        tokenizer=None
    ):
        """
        Initialize CHIT dataset

        Args:
            data_path: Path to CHIT dataset files
            split: Dataset split ('train', 'validation', 'test')
            task_filter: Filter by specific tasks
            culture_filter: Filter by specific cultures
            max_samples: Maximum number of samples to load
            sacred_boundary_protection: Enable sacred content filtering
            tokenizer: Tokenizer for text processing
        """
        self.data_path = Path(data_path)
        self.split = split
        self.task_filter = task_filter
        self.culture_filter = culture_filter
        self.max_samples = max_samples
        self.sacred_boundary_protection = sacred_boundary_protection
        self.tokenizer = tokenizer

        # Load dataset statistics
        self.statistics = CHITStatistics()

        # Initialize task mappings
        self.task_types = {
            "heritage_ner": "Heritage Named Entity Recognition",
            "culture_qa": "Cultural Knowledge Question Answering",
            "story_gen": "Cultural Story Generation",
            "heritage_trans": "Heritage Cross-cultural Translation",
            "ritual_practice": "Ritual Practice Analysis",
            "ecology_know": "Traditional Ecological Knowledge"
        }

        # Load samples
        self.samples = self._load_samples()

        # Create indices for efficient access
        self._create_indices()

    def _load_samples(self) -> List[CHITSample]:
        """Load samples from dataset files"""
        samples_file = self.data_path / f"chit_{self.split}.jsonl"

        if not samples_file.exists():
            # For demo purposes, generate synthetic samples
            return self._generate_demo_samples()

        samples = []
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                sample = CHITSample(**data)

                # Apply filters
                if self.task_filter and sample.task_type not in self.task_filter:
                    continue
                if self.culture_filter and sample.culture not in self.culture_filter:
                    continue

                # Sacred boundary protection
                if self.sacred_boundary_protection and sample.sacred_boundary_flags:
                    continue

                samples.append(sample)

                if self.max_samples and len(samples) >= self.max_samples:
                    break

        return samples

    def _generate_demo_samples(self) -> List[CHITSample]:
        """Generate demo samples for testing"""
        demo_samples = []

        # Heritage NER sample
        demo_samples.append(CHITSample(
            instruction="Extract and classify named entities from this cultural heritage text focusing on ICH-specific categories.",
            input_text="The griot Djeliba sang traditional stories at the harvest festival, using the ancient kora instrument passed down through generations.",
            output="Entities: [griot (MASTER), Djeliba (PERSON), harvest festival (EVENT), kora (INSTRUMENT), generations (TIME_SPAN)]",
            task_type="heritage_ner",
            culture="mandinka",
            language="en",
            cultural_context={"region": "west_africa", "tradition_type": "oral_storytelling"},
            source_dataset="heritage_ner_dataset",
            quality_score=0.92,
            sacred_boundary_flags=[],
            metadata={"annotator": "cultural_expert_001", "validation": "community_approved"}
        ))

        # Cultural QA sample
        demo_samples.append(CHITSample(
            instruction="Answer the following question about traditional cultural practices using authoritative sources.",
            input_text="What is the significance of the number four in Native American spiritual practices?",
            output="The number four holds profound significance in many Native American traditions, representing the four directions (north, south, east, west), the four seasons, the four stages of life, and the four sacred elements. It symbolizes completeness, balance, and the cyclical nature of existence in indigenous cosmology.",
            task_type="culture_qa",
            culture="native_american",
            language="en",
            cultural_context={"domain": "spirituality", "knowledge_type": "sacred_geometry"},
            source_dataset="cultural_qa_dataset",
            quality_score=0.95,
            sacred_boundary_flags=["requires_respect"],
            metadata={"source": "tribal_elder_consultation", "verified": True}
        ))

        # Story generation sample
        demo_samples.append(CHITSample(
            instruction="Generate an authentic traditional narrative following established storytelling conventions.",
            input_text="Create a creation myth explaining the origin of rice cultivation in Chinese culture.",
            output="Long ago, when the heavens and earth were young, the Jade Emperor looked down upon the suffering people who had no food to sustain them. The goddess Shennong took pity and descended to earth. She gathered sacred seeds from the celestial gardens and taught the people to plant them in flooded fields, like mirrors reflecting heaven. Thus rice became the gift that connects earth to sky, nourishing both body and spirit.",
            task_type="story_gen",
            culture="chinese_han",
            language="en",
            cultural_context={"story_type": "creation_myth", "theme": "agricultural_origin"},
            source_dataset="story_generation_dataset",
            quality_score=0.88,
            sacred_boundary_flags=[],
            metadata={"narrative_structure": "traditional_chinese", "cultural_accuracy": "high"}
        ))

        return demo_samples * 100  # Replicate for demo

    def _create_indices(self):
        """Create indices for efficient data access"""
        self.task_index = defaultdict(list)
        self.culture_index = defaultdict(list)
        self.quality_index = defaultdict(list)

        for i, sample in enumerate(self.samples):
            self.task_index[sample.task_type].append(i)
            self.culture_index[sample.culture].append(i)

            # Quality tiers
            if sample.quality_score >= 0.9:
                self.quality_index["high"].append(i)
            elif sample.quality_score >= 0.7:
                self.quality_index["medium"].append(i)
            else:
                self.quality_index["low"].append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = self.samples[idx]

        item = {
            "instruction": sample.instruction,
            "input": sample.input_text,
            "output": sample.output,
            "task_type": sample.task_type,
            "culture": sample.culture,
            "language": sample.language,
            "cultural_context": sample.cultural_context,
            "quality_score": sample.quality_score
        }

        # Tokenize if tokenizer provided
        if self.tokenizer:
            # Create instruction-following format
            full_prompt = f"### Instruction:\n{sample.instruction}\n\n### Input:\n{sample.input_text}\n\n### Response:\n"

            # Tokenize prompt and response
            prompt_tokens = self.tokenizer(
                full_prompt,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )

            response_tokens = self.tokenizer(
                sample.output,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            item.update({
                "input_ids": prompt_tokens["input_ids"].squeeze(),
                "attention_mask": prompt_tokens["attention_mask"].squeeze(),
                "labels": response_tokens["input_ids"].squeeze()
            })

        return item

    def get_samples_by_task(self, task_type: str) -> List[CHITSample]:
        """Get all samples for a specific task"""
        indices = self.task_index.get(task_type, [])
        return [self.samples[i] for i in indices]

    def get_samples_by_culture(self, culture: str) -> List[CHITSample]:
        """Get all samples for a specific culture"""
        indices = self.culture_index.get(culture, [])
        return [self.samples[i] for i in indices]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        actual_stats = {
            "total_samples": len(self.samples),
            "unique_cultures": len(self.culture_index),
            "unique_tasks": len(self.task_index),
            "average_quality": np.mean([s.quality_score for s in self.samples]),
            "task_distribution": {task: len(indices) for task, indices in self.task_index.items()},
            "culture_distribution": {culture: len(indices) for culture, indices in self.culture_index.items()},
            "quality_distribution": {tier: len(indices) for tier, indices in self.quality_index.items()}
        }

        return {
            "design_statistics": self.statistics.__dict__,
            "actual_statistics": actual_stats
        }

    def validate_cultural_authenticity(self, sample_idx: int) -> Dict[str, float]:
        """Validate cultural authenticity of a sample"""
        sample = self.samples[sample_idx]

        # Placeholder for cultural authenticity validation
        # In practice, this would use cultural expert annotations
        authenticity_scores = {
            "cultural_accuracy": 0.9,
            "language_appropriateness": 0.85,
            "context_sensitivity": 0.92,
            "sacred_boundary_respect": 1.0 if not sample.sacred_boundary_flags else 0.8,
            "overall_authenticity": 0.89
        }

        return authenticity_scores


def collate_chit_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for CHIT dataset batching"""

    # Separate tokenized and non-tokenized data
    tokenized_keys = ["input_ids", "attention_mask", "labels"]

    collated = {}

    # Handle tokenized data with padding
    if "input_ids" in batch[0]:
        max_input_len = max(item["input_ids"].size(0) for item in batch)
        max_label_len = max(item["labels"].size(0) for item in batch)

        padded_inputs = []
        padded_masks = []
        padded_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]

            # Pad sequences
            input_padding = max_input_len - input_ids.size(0)
            label_padding = max_label_len - labels.size(0)

            padded_inputs.append(torch.cat([input_ids, torch.zeros(input_padding, dtype=torch.long)]))
            padded_masks.append(torch.cat([attention_mask, torch.zeros(input_padding, dtype=torch.long)]))
            padded_labels.append(torch.cat([labels, torch.full((label_padding,), -100, dtype=torch.long)]))

        collated["input_ids"] = torch.stack(padded_inputs)
        collated["attention_mask"] = torch.stack(padded_masks)
        collated["labels"] = torch.stack(padded_labels)

    # Handle string and metadata
    for key in ["instruction", "input", "output", "task_type", "culture", "language"]:
        collated[key] = [item[key] for item in batch]

    # Handle numerical data
    if "quality_score" in batch[0]:
        collated["quality_score"] = torch.tensor([item["quality_score"] for item in batch])

    return collated
