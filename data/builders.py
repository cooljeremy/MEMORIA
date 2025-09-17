
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from .collectors import CulturalDataSample
from .processors import CulturalPreprocessor, QualityValidator
from .chit_dataset import CHITSample

logger = logging.getLogger(__name__)

@dataclass
class InstructionTemplate:
    """Template for generating instruction-response pairs"""
    task_type: str
    instruction_template: str
    input_format: str
    output_format: str
    cultural_context_required: bool
    examples: List[Dict[str, str]]


class TaskSpecificBuilder:
    """Base class for task-specific CHIT builders"""

    def __init__(self, task_type: str, config: Dict[str, Any]):
        self.task_type = task_type
        self.config = config
        self.templates = self._load_templates()

    def _load_templates(self) -> List[InstructionTemplate]:
        """Load task-specific instruction templates"""
        # This would be loaded from configuration files in practice
        return self._get_default_templates()

    def _get_default_templates(self) -> List[InstructionTemplate]:
        """Get default templates for this task type"""
        return []

    def build_instruction_pairs(
        self,
        raw_samples: List[CulturalDataSample]
    ) -> List[CHITSample]:
        """Convert raw samples to instruction-response pairs"""
        instruction_pairs = []

        for sample in raw_samples:
            try:
                chit_samples = self._process_sample(sample)
                instruction_pairs.extend(chit_samples)
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")

        return instruction_pairs

    def _process_sample(self, sample: CulturalDataSample) -> List[CHITSample]:
        """Process a single raw sample into instruction pairs"""
        raise NotImplementedError("Subclasses must implement _process_sample")


class HeritageNERBuilder(TaskSpecificBuilder):
    """Builder for Heritage Named Entity Recognition task"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("heritage_ner", config)

    def _get_default_templates(self) -> List[InstructionTemplate]:
        return [
            InstructionTemplate(
                task_type="heritage_ner",
                instruction_template="Extract and classify named entities from this cultural heritage text focusing on ICH-specific categories. Identify: (1) Heritage bearers including individual masters, cultural groups, and lineage holders with their titles and affiliations, (2) Geographic locations with cultural significance including sacred sites, practice locations, and transmission centers, (3) Temporal markers including festival dates, seasonal practices, and generational timeframes, (4) Cultural artifacts and their associated practices, (5) Intangible elements including skills, techniques, and knowledge systems. For each entity, provide its category, cultural context, and relationship to other identified entities. Format: 'entity|type|context|relationships'.",
                input_format="Cultural heritage text describing practices, people, places, and artifacts.",
                output_format="List of entities with classifications and relationships.",
                cultural_context_required=True,
                examples=[
                    {
                        "input": "Master weaver Aminata from the Dogon people teaches bogolan techniques.",
                        "output": "Master weaver|TITLE|Dogon cultural context|teaches->bogolan\nAminata|MASTER|Dogon tradition bearer|practices->bogolan\nDogon|ETHNIC_GROUP|West African culture|traditional->weaving\nbogolan|PRACTICE|Mud cloth technique|taught_by->Aminata"
                    }
                ]
            )
        ]

    def _process_sample(self, sample: CulturalDataSample) -> List[CHITSample]:
        """Process sample for Heritage NER task"""
        chit_samples = []

        # Generate entity extraction task
        template = self.templates[0]

        # Extract entities from sample metadata if available
        entities = sample.metadata.get("entities", {})

        if entities:
            # Format entity output
            entity_output = []
            for entity_text, entity_type in entities.items():
                context = f"{sample.culture} cultural context"
                entity_output.append(f"{entity_text}|{entity_type}|{context}|")

            output_text = "\n".join(entity_output)
        else:
            # Generate synthetic entity output based on text analysis
            output_text = self._generate_entity_output(sample.text, sample.culture)

        chit_sample = CHITSample(
            instruction=template.instruction_template,
            input_text=sample.text,
            output=output_text,
            task_type="heritage_ner",
            culture=sample.culture,
            language=sample.language,
            cultural_context=sample.cultural_context,
            source_dataset="heritage_ner",
            quality_score=sample.quality_indicators.get("overall_quality", 0.8),
            sacred_boundary_flags=[],  # Would be determined by sensitivity analysis
            metadata={
                "template_used": "heritage_ner_default",
                "entity_count": len(entities) if entities else 0
            }
        )

        chit_samples.append(chit_sample)
        return chit_samples

    def _generate_entity_output(self, text: str, culture: str) -> str:
        """Generate entity output for text without existing annotations"""
        # Simplified entity detection - in practice would use NER models
        words = text.split()
        entities = []

        # Look for potential person names (capitalized words)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                if i > 0 and words[i-1].lower() in ["master", "elder", "teacher"]:
                    entities.append(f"{word}|MASTER|{culture} tradition bearer|")

        # Look for cultural practices
        cultural_indicators = ["ceremony", "ritual", "dance", "song", "craft", "technique"]
        for word in words:
            if word.lower() in cultural_indicators:
                entities.append(f"{word}|PRACTICE|{culture} cultural practice|")

        return "\n".join(entities) if entities else f"No specific entities identified|GENERAL|{culture} context|"


class CultureQABuilder(TaskSpecificBuilder):
    """Builder for Cultural Knowledge Question Answering task"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("culture_qa", config)

    def _get_default_templates(self) -> List[InstructionTemplate]:
        return [
            InstructionTemplate(
                task_type="culture_qa",
                instruction_template="Based on the provided ethnographic context and community knowledge archives, answer the following question about {cultural_practice}. Your response should: (1) Draw from authoritative community sources and recognized tradition bearers, (2) Acknowledge variations between sub-groups or regional practices, (3) Explain historical evolution and contemporary adaptations, (4) Address common misconceptions or external misrepresentations, (5) Include relevant terminology in the original language with phonetic guides, (6) Respect boundaries around sacred or restricted knowledge.",
                input_format="Question about cultural practices, beliefs, or traditions.",
                output_format="Comprehensive answer with cultural context and sources.",
                cultural_context_required=True,
                examples=[
                    {
                        "input": "What is the significance of the number four in Native American spiritual practices?",
                        "output": "The number four holds profound significance in many Native American traditions, representing the four directions (north, south, east, west), the four seasons, the four stages of life, and the four sacred elements. It symbolizes completeness, balance, and the cyclical nature of existence in indigenous cosmology."
                    }
                ]
            )
        ]

    def _process_sample(self, sample: CulturalDataSample) -> List[CHITSample]:
        """Process sample for Culture QA task"""
        chit_samples = []

        # Generate questions about the cultural content
        questions = self._generate_questions(sample.text, sample.culture)

        template = self.templates[0]

        for question in questions:
            # Generate answer based on sample content
            answer = self._generate_answer(question, sample.text, sample.culture)

            chit_sample = CHITSample(
                instruction=template.instruction_template.format(
                    cultural_practice=f"{sample.culture} cultural practices"
                ),
                input_text=question,
                output=answer,
                task_type="culture_qa",
                culture=sample.culture,
                language=sample.language,
                cultural_context=sample.cultural_context,
                source_dataset="culture_qa",
                quality_score=sample.quality_indicators.get("overall_quality", 0.8),
                sacred_boundary_flags=[],
                metadata={
                    "question_type": "cultural_knowledge",
                    "answer_source": "cultural_text_analysis"
                }
            )

            chit_samples.append(chit_sample)

        return chit_samples

    def _generate_questions(self, text: str, culture: str) -> List[str]:
        """Generate relevant questions about the cultural content"""
        questions = []

        # Question templates based on content type
        if "ceremony" in text.lower() or "ritual" in text.lower():
            questions.extend([
                f"What is the significance of this ceremony in {culture} culture?",
                f"How is this ritual performed in {culture} tradition?",
                f"What are the key elements of this {culture} ceremonial practice?"
            ])

        if "craft" in text.lower() or "technique" in text.lower():
            questions.extend([
                f"How is this traditional craft practiced in {culture} culture?",
                f"What materials and tools are used in this {culture} technique?",
                f"How is this skill transmitted in {culture} communities?"
            ])

        if "story" in text.lower() or "legend" in text.lower():
            questions.extend([
                f"What is the cultural meaning behind this {culture} story?",
                f"How do such narratives function in {culture} society?",
                f"What values does this story convey in {culture} culture?"
            ])

        # Default questions if no specific patterns found
        if not questions:
            questions = [
                f"What can you tell me about this aspect of {culture} culture?",
                f"How does this practice relate to {culture} traditions?",
                f"What is the cultural significance of this in {culture} heritage?"
            ]

        return questions[:2]  # Return top 2 questions

    def _generate_answer(self, question: str, text: str, culture: str) -> str:
        """Generate culturally-informed answer"""
        # Extract key information from the text
        key_phrases = [phrase.strip() for phrase in text.split('.') if phrase.strip()]

        # Build answer incorporating cultural context
        answer_parts = []

        # Add cultural context
        answer_parts.append(f"In {culture} culture, ")

        # Use relevant content from text
        relevant_content = [phrase for phrase in key_phrases if len(phrase) > 20]
        if relevant_content:
            answer_parts.append(relevant_content[0].lower())
        else:
            answer_parts.append("this practice represents an important cultural tradition")

        # Add cultural significance
        answer_parts.append(f". This reflects the broader cultural values and worldview of {culture} communities.")

        # Add respect for cultural boundaries
        answer_parts.append(" It's important to note that some aspects of this tradition may have restricted access or require community permission to fully understand.")

        return "".join(answer_parts)


class StoryGenBuilder(TaskSpecificBuilder):
    """Builder for Cultural Story Generation task"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("story_gen", config)

    def _get_default_templates(self) -> List[InstructionTemplate]:
        return [
            InstructionTemplate(
                task_type="story_gen",
                instruction_template="Generate an authentic traditional narrative following the established storytelling conventions of {culture} tradition. The story should address the theme of {theme} while incorporating: (1) Appropriate opening and closing formulas specific to {culture} oral tradition, (2) Culture-specific narrative devices (repetition patterns, number symbolism, directional significance), (3) Traditional character archetypes and their expected behavioral patterns, (4) Incorporation of relevant proverbs, riddles, or songs as per custom, (5) Appropriate moral resolution aligned with community values. Ensure the narrative respects cultural taboos and sacred knowledge restrictions.",
                input_format="Theme or prompt for story generation with cultural specifications.",
                output_format="Complete traditional narrative following cultural conventions.",
                cultural_context_required=True,
                examples=[
                    {
                        "input": "Create a creation myth explaining the origin of rice cultivation in Chinese culture.",
                        "output": "Long ago, when the heavens and earth were young, the Jade Emperor looked down upon the suffering people who had no food to sustain them. The goddess Shennong took pity and descended to earth..."
                    }
                ]
            )
        ]

    def _process_sample(self, sample: CulturalDataSample) -> List[CHITSample]:
        """Process sample for Story Generation task"""
        chit_samples = []

        # Extract story themes from the sample
        themes = self._extract_story_themes(sample.text, sample.culture)

        template = self.templates[0]

        for theme in themes:
            # Generate story prompt
            story_prompt = f"Create a {theme} story in the {sample.culture} tradition"

            # Generate story based on sample content and cultural patterns
            story = self._generate_story(theme, sample.culture, sample.text)

            chit_sample = CHITSample(
                instruction=template.instruction_template.format(
                    culture=sample.culture,
                    theme=theme
                ),
                input_text=story_prompt,
                output=story,
                task_type="story_gen",
                culture=sample.culture,
                language=sample.language,
                cultural_context=sample.cultural_context,
                source_dataset="story_gen",
                quality_score=sample.quality_indicators.get("overall_quality", 0.8),
                sacred_boundary_flags=[],
                metadata={
                    "story_theme": theme,
                    "narrative_style": f"{sample.culture}_traditional"
                }
            )

            chit_samples.append(chit_sample)

        return chit_samples

    def _extract_story_themes(self, text: str, culture: str) -> List[str]:
        """Extract potential story themes from cultural text"""
        themes = []

        # Common cultural story themes
        theme_indicators = {
            "origin": ["origin", "beginning", "first", "creation", "started"],
            "wisdom": ["teaching", "lesson", "wisdom", "knowledge", "learned"],
            "heroic": ["brave", "hero", "courage", "quest", "journey"],
            "moral": ["right", "wrong", "virtue", "honor", "respect"],
            "nature": ["animal", "tree", "river", "mountain", "earth"],
            "spiritual": ["spirit", "sacred", "divine", "blessing", "prayer"]
        }

        text_lower = text.lower()
        for theme, indicators in theme_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                themes.append(theme)

        return themes if themes else ["traditional_wisdom"]

    def _generate_story(self, theme: str, culture: str, reference_text: str) -> str:
        """Generate story following cultural conventions"""

        # Cultural opening formulas
        openings = {
            "chinese_han": "Long ago, when the heavens and earth were young",
            "japanese": "In the time of our ancestors, when spirits walked among us",
            "native_american": "In the old days, when animals could speak",
            "african_traditional": "Listen, children of the village, to this story of old",
            "hindu": "In ancient times, when gods and mortals lived closer together",
            "generic": "In times long past, when the world was different"
        }

        # Cultural closing formulas
        closings = {
            "chinese_han": "And so harmony was restored between heaven and earth",
            "japanese": "This is why we honor the old ways to this day",
            "native_american": "And that is how things came to be as they are",
            "african_traditional": "This story has been told by our people for generations",
            "hindu": "Thus did the divine order maintain balance in the world",
            "generic": "And the wisdom of this story lives on"
        }

        opening = openings.get(culture, openings["generic"])
        closing = closings.get(culture, closings["generic"])

        # Generate story content based on theme and reference
        story_content = self._generate_story_content(theme, culture, reference_text)

        return f"{opening}, {story_content}. {closing}."

    def _generate_story_content(self, theme: str, culture: str, reference_text: str) -> str:
        """Generate the main story content"""

        # Extract key elements from reference text
        key_elements = [elem.strip() for elem in reference_text.split('.') if elem.strip()]

        if theme == "origin":
            content = f"there came a time when the people needed to understand the origins of their most sacred traditions"
        elif theme == "wisdom":
            content = f"a wise elder shared knowledge that would guide the community through difficult times"
        elif theme == "heroic":
            content = f"a brave soul undertook a journey to protect what the community held most dear"
        elif theme == "moral":
            content = f"the community learned the importance of living in harmony with their values and traditions"
        elif theme == "nature":
            content = f"the natural world revealed its secrets to those who approached with respect and understanding"
        else:
            content = f"the people discovered the true meaning of their cultural heritage"

        # Incorporate elements from reference text if appropriate
        if key_elements and len(key_elements[0]) > 10:
            content += f", and {key_elements[0].lower()}"

        return content


class CHITDatasetBuilder:
    """
    Main builder for the complete CHIT dataset

    Orchestrates all task-specific builders to create the 158K sample
    Cultural Heritage Instruction Tuning dataset.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cultural_processor = CulturalPreprocessor(config.get("preprocessing", {}))
        self.quality_validator = QualityValidator(config.get("quality_thresholds", {}))

        # Initialize task-specific builders
        self.task_builders = {
            "heritage_ner": HeritageNERBuilder(config.get("heritage_ner", {})),
            "culture_qa": CultureQABuilder(config.get("culture_qa", {})),
            "story_gen": StoryGenBuilder(config.get("story_gen", {}))
            # Add other builders as needed
        }

    def build_chit_dataset(
        self,
        raw_samples: Dict[str, List[CulturalDataSample]],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Build complete CHIT dataset from raw cultural samples

        Args:
            raw_samples: Raw samples organized by source
            output_path: Path to save the dataset

        Returns:
            Build statistics and metadata
        """

        logger.info("Starting CHIT dataset construction...")

        all_chit_samples = []
        build_stats = {
            "total_samples": 0,
            "samples_by_task": {},
            "samples_by_culture": {},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "sacred_content_filtered": 0
        }

        # Process each data source
        for source_name, source_samples in raw_samples.items():
            logger.info(f"Processing {len(source_samples)} samples from {source_name}")

            # Determine which tasks to generate for this source
            applicable_tasks = self._determine_applicable_tasks(source_name)

            for task_type in applicable_tasks:
                if task_type in self.task_builders:
                    builder = self.task_builders[task_type]
                    task_samples = builder.build_instruction_pairs(source_samples)

                    # Quality validation and filtering
                    validated_samples = []
                    for sample in task_samples:
                        # Cultural preprocessing
                        processed = self.cultural_processor.preprocess_cultural_text(
                            sample.input_text,
                            sample.culture
                        )

                        # Apply sacred boundary protection
                        if processed["sensitivity_info"].sensitivity_level.value in ["sacred", "forbidden"]:
                            build_stats["sacred_content_filtered"] += 1
                            continue

                        # Quality validation
                        quality_scores = self.quality_validator.validate_cultural_sample(
                            sample.input_text,
                            sample.culture,
                            sample.metadata
                        )

                        if self.quality_validator.meets_quality_threshold(quality_scores):
                            sample.quality_score = quality_scores["overall_quality"]
                            validated_samples.append(sample)

                            # Update statistics
                            if quality_scores["overall_quality"] >= 0.9:
                                build_stats["quality_distribution"]["high"] += 1
                            elif quality_scores["overall_quality"] >= 0.7:
                                build_stats["quality_distribution"]["medium"] += 1
                            else:
                                build_stats["quality_distribution"]["low"] += 1

                    all_chit_samples.extend(validated_samples)

                    # Update task statistics
                    if task_type not in build_stats["samples_by_task"]:
                        build_stats["samples_by_task"][task_type] = 0
                    build_stats["samples_by_task"][task_type] += len(validated_samples)

        # Update culture statistics
        for sample in all_chit_samples:
            if sample.culture not in build_stats["samples_by_culture"]:
                build_stats["samples_by_culture"][sample.culture] = 0
            build_stats["samples_by_culture"][sample.culture] += 1

        build_stats["total_samples"] = len(all_chit_samples)

        # Save dataset
        self._save_chit_dataset(all_chit_samples, output_path, build_stats)

        logger.info(f"CHIT dataset construction complete: {build_stats['total_samples']} samples")
        return build_stats

    def _determine_applicable_tasks(self, source_name: str) -> List[str]:
        """Determine which tasks are applicable for a data source"""

        task_mapping = {
            "unesco": ["heritage_ner", "culture_qa"],
            "heritage_ner": ["heritage_ner"],
            "craft_knowledge": ["heritage_ner", "culture_qa", "story_gen"],
            "ritual_practice": ["heritage_ner", "culture_qa"],
            "music_dance": ["heritage_ner", "culture_qa", "story_gen"],
            "ecology_knowledge": ["culture_qa", "story_gen"]
        }

        return task_mapping.get(source_name, ["culture_qa"])

    def _save_chit_dataset(
        self,
        samples: List[CHITSample],
        output_path: str,
        build_stats: Dict[str, Any]
    ):
        """Save the CHIT dataset to files"""

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split dataset
        random.shuffle(samples)
        train_size = int(0.8 * len(samples))
        val_size = int(0.1 * len(samples))

        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]

        # Save splits
        splits = {
            "train": train_samples,
            "validation": val_samples,
            "test": test_samples
        }

        for split_name, split_samples in splits.items():
            split_file = output_dir / f"chit_{split_name}.jsonl"
            with open(split_file, 'w', encoding='utf-8') as f:
                for sample in split_samples:
                    f.write(json.dumps(sample.__dict__, ensure_ascii=False) + '\n')

            logger.info(f"Saved {len(split_samples)} samples to {split_file}")

        # Save build statistics
        stats_file = output_dir / "build_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(build_stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Dataset statistics saved to {stats_file}")


# Example usage and testing
if __name__ == "__main__":
    # Demo configuration
    config = {
        "preprocessing": {
            "cultural_sensitivity_threshold": 0.8
        },
        "quality_thresholds": {
            "overall_quality": 0.7
        }
    }

    # Initialize builder
    builder = CHITDatasetBuilder(config)

    # This would be called with real collected data
    logger.info("CHIT Dataset Builder initialized successfully")