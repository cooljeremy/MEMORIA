
import re
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


class BaseTaskEvaluator(ABC):
    """Base class for task-specific evaluators."""

    @abstractmethod
    def evaluate(self,
                prediction: str,
                reference: str,
                cultural_context: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate prediction against reference with cultural context."""
        pass


class HeritageNEREvaluator(BaseTaskEvaluator):
    """Evaluator for Heritage Named Entity Recognition task."""

    def __init__(self):
        # Define cultural heritage entity types
        self.entity_types = [
            "PERSON", "LOCATION", "TRADITION", "ARTIFACT", "PRACTICE",
            "EVENT", "ORGANIZATION", "CULTURAL_CONCEPT", "TIME_PERIOD"
        ]

    def evaluate(self,
                prediction: str,
                reference: str,
                cultural_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate NER predictions using precision, recall, and F1 scores.

        Expected format: "ENTITY_TEXT [ENTITY_TYPE]"
        """
        pred_entities = self._extract_entities(prediction)
        ref_entities = self._extract_entities(reference)

        # Calculate exact match metrics
        exact_precision, exact_recall, exact_f1 = self._calculate_exact_match_metrics(
            pred_entities, ref_entities
        )

        # Calculate partial match metrics (for cultural entity variations)
        partial_precision, partial_recall, partial_f1 = self._calculate_partial_match_metrics(
            pred_entities, ref_entities
        )

        # Calculate type accuracy
        type_accuracy = self._calculate_type_accuracy(pred_entities, ref_entities)

        # Cultural entity bonus (reward culturally appropriate entities)
        cultural_bonus = self._calculate_cultural_entity_bonus(
            pred_entities, cultural_context.get("culture", "")
        )

        return {
            "exact_precision": exact_precision,
            "exact_recall": exact_recall,
            "exact_f1": exact_f1,
            "partial_precision": partial_precision,
            "partial_recall": partial_recall,
            "partial_f1": partial_f1,
            "type_accuracy": type_accuracy,
            "cultural_entity_bonus": cultural_bonus,
            "overall_score": (exact_f1 * 0.4 + partial_f1 * 0.3 +
                            type_accuracy * 0.2 + cultural_bonus * 0.1)
        }

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities from text in format 'ENTITY [TYPE]'."""
        entities = []
        # Pattern to match: entity_text [ENTITY_TYPE]
        pattern = r'(.+?)\s*\[([A-Z_]+)\]'
        matches = re.findall(pattern, text)

        for entity_text, entity_type in matches:
            entity_text = entity_text.strip()
            entity_type = entity_type.strip().upper()
            if entity_type in self.entity_types:
                entities.append((entity_text, entity_type))

        return entities

    def _calculate_exact_match_metrics(self,
                                     pred_entities: List[Tuple[str, str]],
                                     ref_entities: List[Tuple[str, str]]) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 for exact entity matches."""
        if not ref_entities:
            return 1.0 if not pred_entities else 0.0, 1.0, 1.0 if not pred_entities else 0.0

        pred_set = set(pred_entities)
        ref_set = set(ref_entities)

        true_positives = len(pred_set.intersection(ref_set))
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(ref_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def _calculate_partial_match_metrics(self,
                                       pred_entities: List[Tuple[str, str]],
                                       ref_entities: List[Tuple[str, str]]) -> Tuple[float, float, float]:
        """Calculate metrics for partial entity matches (same type, overlapping text)."""
        if not ref_entities:
            return 1.0 if not pred_entities else 0.0, 1.0, 1.0 if not pred_entities else 0.0

        partial_matches = 0

        for pred_text, pred_type in pred_entities:
            for ref_text, ref_type in ref_entities:
                if pred_type == ref_type:
                    # Check text similarity
                    pred_words = set(pred_text.lower().split())
                    ref_words = set(ref_text.lower().split())
                    if pred_words.intersection(ref_words):
                        partial_matches += 1
                        break

        precision = partial_matches / len(pred_entities) if pred_entities else 0.0
        recall = partial_matches / len(ref_entities)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def _calculate_type_accuracy(self,
                               pred_entities: List[Tuple[str, str]],
                               ref_entities: List[Tuple[str, str]]) -> float:
        """Calculate accuracy of entity type classification."""
        if not ref_entities:
            return 1.0

        type_matches = 0
        total_comparisons = 0

        for ref_text, ref_type in ref_entities:
            for pred_text, pred_type in pred_entities:
                # If texts are similar, check type accuracy
                ref_words = set(ref_text.lower().split())
                pred_words = set(pred_text.lower().split())
                if ref_words.intersection(pred_words):
                    total_comparisons += 1
                    if pred_type == ref_type:
                        type_matches += 1
                    break

        return type_matches / total_comparisons if total_comparisons > 0 else 0.0

    def _calculate_cultural_entity_bonus(self,
                                       pred_entities: List[Tuple[str, str]],
                                       culture: str) -> float:
        """Calculate bonus for culturally appropriate entity recognition."""
        if not culture or not pred_entities:
            return 0.0

        cultural_entities = 0
        for entity_text, entity_type in pred_entities:
            if entity_type in ["TRADITION", "PRACTICE", "CULTURAL_CONCEPT", "ARTIFACT"]:
                # Simple heuristic: check if entity text contains culture-specific terms
                if culture.lower() in entity_text.lower():
                    cultural_entities += 1

        return min(1.0, cultural_entities / len(pred_entities))


class CultureQAEvaluator(BaseTaskEvaluator):
    """Evaluator for Cultural Question Answering task."""

    def evaluate(self,
                prediction: str,
                reference: str,
                cultural_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate QA predictions using answer similarity and cultural accuracy.
        """
        # Answer similarity (simplified)
        answer_similarity = self._calculate_answer_similarity(prediction, reference)

        # Factual accuracy
        factual_accuracy = self._calculate_factual_accuracy(
            prediction, reference, cultural_context
        )

        # Cultural relevance
        cultural_relevance = self._calculate_cultural_relevance(
            prediction, cultural_context.get("culture", "")
        )

        # Answer completeness
        completeness = self._calculate_completeness(prediction, reference)

        return {
            "answer_similarity": answer_similarity,
            "factual_accuracy": factual_accuracy,
            "cultural_relevance": cultural_relevance,
            "completeness": completeness,
            "overall_score": (answer_similarity * 0.3 + factual_accuracy * 0.3 +
                            cultural_relevance * 0.2 + completeness * 0.2)
        }

    def _calculate_answer_similarity(self, prediction: str, reference: str) -> float:
        """Calculate similarity between predicted and reference answers."""
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())

        if not ref_words:
            return 1.0 if not pred_words else 0.0

        intersection = pred_words.intersection(ref_words)
        union = pred_words.union(ref_words)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_factual_accuracy(self,
                                  prediction: str,
                                  reference: str,
                                  cultural_context: Dict[str, Any]) -> float:
        """Calculate factual accuracy of the answer."""
        # Simplified factual checking - in practice would use fact databases
        pred_lower = prediction.lower()

        # Check for common factual indicators
        accurate_patterns = [
            r"\baccording to\b", r"\btraditionally\b", r"\bhistorically\b",
            r"\bin.*culture\b", r"\bis.*known\b"
        ]

        inaccurate_patterns = [
            r"\bi think\b", r"\bmaybe\b", r"\bprobably\b", r"\bnot sure\b",
            r"\bunsure\b", r"\bguess\b"
        ]

        accuracy_score = 0.5  # Base score

        for pattern in accurate_patterns:
            if re.search(pattern, pred_lower):
                accuracy_score += 0.1

        for pattern in inaccurate_patterns:
            if re.search(pattern, pred_lower):
                accuracy_score -= 0.15

        return max(0.0, min(1.0, accuracy_score))

    def _calculate_cultural_relevance(self, prediction: str, culture: str) -> float:
        """Calculate how culturally relevant the answer is."""
        if not culture:
            return 0.5

        pred_lower = prediction.lower()
        culture_lower = culture.lower()

        # Check for culture-specific mentions
        relevance_score = 0.0

        if culture_lower in pred_lower:
            relevance_score += 0.3

        # Check for cultural concepts
        cultural_terms = [
            "tradition", "custom", "practice", "heritage", "culture",
            "ritual", "ceremony", "festival", "belief", "value"
        ]

        for term in cultural_terms:
            if term in pred_lower:
                relevance_score += 0.1

        return min(1.0, relevance_score)

    def _calculate_completeness(self, prediction: str, reference: str) -> float:
        """Calculate how complete the answer is."""
        pred_sentences = len([s for s in prediction.split('.') if s.strip()])
        ref_sentences = len([s for s in reference.split('.') if s.strip()])

        if ref_sentences == 0:
            return 1.0 if pred_sentences == 0 else 0.0

        # Penalize answers that are too short or too long
        length_ratio = pred_sentences / ref_sentences
        if 0.7 <= length_ratio <= 1.5:
            return 1.0
        elif length_ratio < 0.3:
            return 0.3  # Too short
        elif length_ratio > 3.0:
            return 0.5  # Too long
        else:
            return 0.7


class StoryGenerationEvaluator(BaseTaskEvaluator):
    """Evaluator for culturally authentic story generation."""

    def evaluate(self,
                prediction: str,
                reference: str,
                cultural_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate story generation for cultural authenticity and narrative quality.
        """
        # Narrative structure
        narrative_quality = self._calculate_narrative_quality(prediction)

        # Cultural authenticity
        cultural_authenticity = self._calculate_cultural_authenticity(
            prediction, cultural_context
        )

        # Character development
        character_development = self._calculate_character_development(prediction)

        # Cultural message/theme
        cultural_theme = self._calculate_cultural_theme(
            prediction, cultural_context.get("culture", "")
        )

        # Language appropriateness
        language_appropriateness = self._calculate_language_appropriateness(
            prediction, cultural_context
        )

        return {
            "narrative_quality": narrative_quality,
            "cultural_authenticity": cultural_authenticity,
            "character_development": character_development,
            "cultural_theme": cultural_theme,
            "language_appropriateness": language_appropriateness,
            "overall_score": (narrative_quality * 0.25 + cultural_authenticity * 0.3 +
                            character_development * 0.15 + cultural_theme * 0.2 +
                            language_appropriateness * 0.1)
        }

    def _calculate_narrative_quality(self, story: str) -> float:
        """Calculate basic narrative structure quality."""
        sentences = [s.strip() for s in story.split('.') if s.strip()]

        if len(sentences) < 3:
            return 0.3  # Too short for a story

        # Check for story elements
        story_lower = story.lower()
        story_elements = [
            r"\bonce upon a time\b", r"\bin the beginning\b", r"\blong ago\b",
            r"\bthen\b", r"\bnext\b", r"\bfinally\b", r"\bin the end\b",
            r"\bcharacter\b", r"\bprotagonist\b", r"\bhero\b", r"\bmain.*character\b"
        ]

        element_score = sum(1 for pattern in story_elements if re.search(pattern, story_lower))
        return min(1.0, 0.5 + element_score * 0.1)

    def _calculate_cultural_authenticity(self,
                                       story: str,
                                       cultural_context: Dict[str, Any]) -> float:
        """Calculate cultural authenticity of the story."""
        culture = cultural_context.get("culture", "")
        domain = cultural_context.get("domain", "")

        if not culture:
            return 0.5

        story_lower = story.lower()
        authenticity_score = 0.0

        # Check for cultural elements
        if culture.lower() in story_lower:
            authenticity_score += 0.2

        # Check for domain-specific elements
        domain_elements = {
            "oral_traditions": ["legend", "myth", "folktale", "proverb", "wisdom"],
            "performing_arts": ["dance", "music", "performance", "art", "song"],
            "social_practices": ["ceremony", "festival", "ritual", "celebration", "community"],
            "traditional_craftsmanship": ["craft", "artisan", "skill", "technique", "creation"],
            "knowledge_practices": ["knowledge", "wisdom", "teaching", "learning", "tradition"]
        }

        if domain in domain_elements:
            for element in domain_elements[domain]:
                if element in story_lower:
                    authenticity_score += 0.1

        # Check for cultural values
        cultural_values = [
            "respect", "honor", "family", "community", "tradition",
            "wisdom", "harmony", "balance", "ancestor", "heritage"
        ]

        for value in cultural_values:
            if value in story_lower:
                authenticity_score += 0.05

        return min(1.0, authenticity_score)

    def _calculate_character_development(self, story: str) -> float:
        """Calculate character development quality."""
        story_lower = story.lower()

        character_indicators = [
            r"\bhe\b", r"\bshe\b", r"\bthey\b", r"\bcharacter\b",
            r"\bprotagonist\b", r"\bhero\b", r"\bmain.*character\b",
            r"\byoung.*\w+\b", r"\bold.*\w+\b", r"\bwise.*\w+\b"
        ]

        development_indicators = [
            r"\blearned\b", r"\bgrew\b", r"\bchanged\b", r"\brealized\b",
            r"\bunderstood\b", r"\bbecame\b", r"\bdeveloped\b", r"\bmatured\b"
        ]

        character_score = sum(1 for pattern in character_indicators if re.search(pattern, story_lower))
        development_score = sum(1 for pattern in development_indicators if re.search(pattern, story_lower))

        if character_score == 0:
            return 0.2

        return min(1.0, 0.3 + (character_score + development_score) * 0.1)

    def _calculate_cultural_theme(self, story: str, culture: str) -> float:
        """Calculate presence and relevance of cultural themes."""
        if not culture:
            return 0.5

        story_lower = story.lower()

        cultural_themes = [
            "tradition", "heritage", "ancestor", "wisdom", "respect",
            "community", "family", "honor", "duty", "harmony",
            "balance", "nature", "spirit", "belief", "value"
        ]

        theme_score = 0.0
        for theme in cultural_themes:
            if theme in story_lower:
                theme_score += 0.1

        # Bonus for culture-specific themes
        if culture.lower() in story_lower:
            theme_score += 0.2

        return min(1.0, theme_score)

    def _calculate_language_appropriateness(self,
                                          story: str,
                                          cultural_context: Dict[str, Any]) -> float:
        """Calculate appropriateness of language for the cultural context."""
        story_lower = story.lower()

        # Check for appropriate tone
        appropriate_tone = [
            r"\breverently\b", r"\brespectfully\b", r"\bcarefully\b",
            r"\bwisely\b", r"\btradition\w*\b", r"\bancient\b"
        ]

        inappropriate_tone = [
            r"\bweird\b", r"\bstrange\b", r"\bodd\b", r"\bbizarre\b",
            r"\bprimitive\b", r"\bbackward\b"
        ]

        appropriateness_score = 0.5

        for pattern in appropriate_tone:
            if re.search(pattern, story_lower):
                appropriateness_score += 0.1

        for pattern in inappropriate_tone:
            if re.search(pattern, story_lower):
                appropriateness_score -= 0.15

        return max(0.0, min(1.0, appropriateness_score))


class TranslationEvaluator(BaseTaskEvaluator):
    """Evaluator for cross-cultural translation task."""

    def evaluate(self,
                prediction: str,
                reference: str,
                cultural_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate translation quality with cultural considerations.
        """
        # Translation accuracy (simplified BLEU-like score)
        translation_accuracy = self._calculate_translation_accuracy(prediction, reference)

        # Fluency
        fluency = self._calculate_fluency(prediction)

        # Cultural adaptation
        cultural_adaptation = self._calculate_cultural_adaptation(
            prediction, cultural_context
        )

        # Meaning preservation
        meaning_preservation = self._calculate_meaning_preservation(
            prediction, reference, cultural_context
        )

        return {
            "translation_accuracy": translation_accuracy,
            "fluency": fluency,
            "cultural_adaptation": cultural_adaptation,
            "meaning_preservation": meaning_preservation,
            "overall_score": (translation_accuracy * 0.3 + fluency * 0.2 +
                            cultural_adaptation * 0.3 + meaning_preservation * 0.2)
        }

    def _calculate_translation_accuracy(self, prediction: str, reference: str) -> float:
        """Calculate translation accuracy using n-gram overlap."""
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()

        if not ref_words:
            return 1.0 if not pred_words else 0.0

        # Simplified BLEU-1 (unigram precision)
        matching_words = 0
        for word in pred_words:
            if word in ref_words:
                matching_words += 1

        precision = matching_words / len(pred_words) if pred_words else 0.0
        return min(1.0, precision * 1.2)  # Slight boost for reasonable translations

    def _calculate_fluency(self, translation: str) -> float:
        """Calculate fluency of the translation."""
        # Simple heuristics for fluency
        sentences = [s.strip() for s in translation.split('.') if s.strip()]

        if not sentences:
            return 0.0

        fluency_score = 0.7  # Base score

        # Check for grammatical issues (simplified)
        grammar_issues = [
            r'\b(a)\s+(a)\b',  # Double articles
            r'\b(the)\s+(the)\b',  # Double determiners
            r'\w+ing\s+\w+ing',  # Double -ing verbs (potential issue)
        ]

        translation_lower = translation.lower()
        for pattern in grammar_issues:
            if re.search(pattern, translation_lower):
                fluency_score -= 0.1

        # Reward natural language patterns
        natural_patterns = [
            r'\b(and|but|however|therefore|moreover)\b',  # Connectors
            r'\b(because|since|although|while)\b',  # Subordinators
        ]

        for pattern in natural_patterns:
            if re.search(pattern, translation_lower):
                fluency_score += 0.05

        return max(0.0, min(1.0, fluency_score))

    def _calculate_cultural_adaptation(self,
                                     translation: str,
                                     cultural_context: Dict[str, Any]) -> float:
        """Calculate cultural adaptation quality."""
        target_culture = cultural_context.get("target_culture", "")

        if not target_culture:
            return 0.5

        translation_lower = translation.lower()
        target_culture_lower = target_culture.lower()

        adaptation_score = 0.0

        # Check for target culture mentions
        if target_culture_lower in translation_lower:
            adaptation_score += 0.3

        # Check for cultural adaptation indicators
        adaptation_indicators = [
            r"\bin.*culture\b", r"\badapted.*for\b", r"\blocalized\b",
            r"\bcultural.*context\b", r"\bappropriate.*for\b"
        ]

        for pattern in adaptation_indicators:
            if re.search(pattern, translation_lower):
                adaptation_score += 0.1

        return min(1.0, adaptation_score)

    def _calculate_meaning_preservation(self,
                                      translation: str,
                                      reference: str,
                                      cultural_context: Dict[str, Any]) -> float:
        """Calculate how well the core meaning is preserved."""
        # Extract key concepts from reference
        ref_words = set(reference.lower().split())
        trans_words = set(translation.lower().split())

        # Calculate concept overlap
        concept_overlap = len(ref_words.intersection(trans_words))
        total_concepts = len(ref_words)

        if total_concepts == 0:
            return 1.0

        base_preservation = concept_overlap / total_concepts

        # Adjust for cultural context
        source_culture = cultural_context.get("source_culture", "")
        if source_culture and source_culture.lower() in reference.lower():
            # Source culture should be preserved or adapted
            if (source_culture.lower() in translation.lower() or
                cultural_context.get("target_culture", "").lower() in translation.lower()):
                base_preservation += 0.1

        return min(1.0, base_preservation)


class ClassificationEvaluator(BaseTaskEvaluator):
    """Evaluator for cultural classification task."""

    def __init__(self):
        self.domains = [
            "oral_traditions", "performing_arts", "social_practices",
            "traditional_craftsmanship", "knowledge_practices", "other"
        ]

    def evaluate(self,
                prediction: str,
                reference: str,
                cultural_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate classification predictions.
        """
        # Extract predicted labels
        pred_labels = self._extract_labels(prediction)
        ref_labels = self._extract_labels(reference)

        # Calculate accuracy
        accuracy = self._calculate_accuracy(pred_labels, ref_labels)

        # Calculate precision, recall, F1
        precision, recall, f1 = self._calculate_prf_metrics(pred_labels, ref_labels)

        # Domain-specific accuracy
        domain_accuracy = self._calculate_domain_accuracy(pred_labels, ref_labels)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "domain_accuracy": domain_accuracy,
            "overall_score": (accuracy * 0.4 + f1 * 0.4 + domain_accuracy * 0.2)
        }

    def _extract_labels(self, text: str) -> Set[str]:
        """Extract classification labels from text."""
        text_lower = text.lower()
        labels = set()

        # Look for domain labels
        for domain in self.domains:
            if domain.replace('_', ' ') in text_lower or domain in text_lower:
                labels.add(domain)

        # If no specific domains found, look for general terms
        if not labels:
            general_terms = {
                "story": "oral_traditions",
                "tale": "oral_traditions",
                "legend": "oral_traditions",
                "music": "performing_arts",
                "dance": "performing_arts",
                "art": "performing_arts",
                "festival": "social_practices",
                "ceremony": "social_practices",
                "ritual": "social_practices",
                "craft": "traditional_craftsmanship",
                "skill": "traditional_craftsmanship",
                "technique": "traditional_craftsmanship",
                "knowledge": "knowledge_practices",
                "wisdom": "knowledge_practices",
                "medicine": "knowledge_practices"
            }

            for term, domain in general_terms.items():
                if term in text_lower:
                    labels.add(domain)

        return labels if labels else {"other"}

    def _calculate_accuracy(self, pred_labels: Set[str], ref_labels: Set[str]) -> float:
        """Calculate classification accuracy."""
        if not ref_labels:
            return 1.0 if not pred_labels else 0.0

        # For multi-label, use Jaccard similarity
        intersection = pred_labels.intersection(ref_labels)
        union = pred_labels.union(ref_labels)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_prf_metrics(self, pred_labels: Set[str], ref_labels: Set[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if not ref_labels:
            return 1.0 if not pred_labels else 0.0, 1.0, 1.0 if not pred_labels else 0.0

        true_positives = len(pred_labels.intersection(ref_labels))
        precision = true_positives / len(pred_labels) if pred_labels else 0.0
        recall = true_positives / len(ref_labels) if ref_labels else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def _calculate_domain_accuracy(self, pred_labels: Set[str], ref_labels: Set[str]) -> float:
        """Calculate accuracy specifically for UNESCO domain classification."""
        pred_domains = pred_labels.intersection(set(self.domains))
        ref_domains = ref_labels.intersection(set(self.domains))

        if not ref_domains:
            return 1.0 if not pred_domains else 0.0

        return len(pred_domains.intersection(ref_domains)) / len(ref_domains.union(pred_domains))


class DialogueEvaluator(BaseTaskEvaluator):
    """Evaluator for cultural context-aware dialogue."""

    def evaluate(self,
                prediction: str,
                reference: str,
                cultural_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate dialogue generation for cultural appropriateness and relevance.
        """
        # Response appropriateness
        appropriateness = self._calculate_response_appropriateness(
            prediction, cultural_context
        )

        # Cultural sensitivity
        cultural_sensitivity = self._calculate_cultural_sensitivity(
            prediction, cultural_context.get("culture", "")
        )

        # Dialogue coherence
        coherence = self._calculate_coherence(prediction, reference)

        # Informativeness
        informativeness = self._calculate_informativeness(prediction, cultural_context)

        return {
            "appropriateness": appropriateness,
            "cultural_sensitivity": cultural_sensitivity,
            "coherence": coherence,
            "informativeness": informativeness,
            "overall_score": (appropriateness * 0.3 + cultural_sensitivity * 0.3 +
                            coherence * 0.2 + informativeness * 0.2)
        }

    def _calculate_response_appropriateness(self,
                                          response: str,
                                          cultural_context: Dict[str, Any]) -> float:
        """Calculate appropriateness of response given cultural context."""
        response_lower = response.lower()

        # Check for appropriate tone indicators
        appropriate_indicators = [
            r"\brespectfully\b", r"\bwith respect\b", r"\bhonorably\b",
            r"\btraditionally\b", r"\bin.*culture\b", r"\baccording to\b"
        ]

        inappropriate_indicators = [
            r"\bweird\b", r"\bstrange\b", r"\bodd\b", r"\bprimitive\b",
            r"\bbackward\b", r"\buncivilized\b"
        ]

        appropriateness_score = 0.5

        for pattern in appropriate_indicators:
            if re.search(pattern, response_lower):
                appropriateness_score += 0.1

        for pattern in inappropriate_indicators:
            if re.search(pattern, response_lower):
                appropriateness_score -= 0.2

        return max(0.0, min(1.0, appropriateness_score))

    def _calculate_cultural_sensitivity(self, response: str, culture: str) -> float:
        """Calculate cultural sensitivity of the response."""
        if not culture:
            return 0.5

        response_lower = response.lower()

        # Check for cultural sensitivity indicators
        sensitivity_indicators = [
            r"\bunderstand\b", r"\brespect\b", r"\bhonor\b", r"\bvalue\b",
            r"\bappreciate\b", r"\backnowledge\b", r"\baware.*of\b"
        ]

        insensitivity_indicators = [
            r"\bjust\b.*\bbelief\b", r"\bonly\b.*\btradition\b",
            r"\bsuperstition\b", r"\bmyth\b.*\bfalse\b"
        ]

        sensitivity_score = 0.5

        for pattern in sensitivity_indicators:
            if re.search(pattern, response_lower):
                sensitivity_score += 0.1

        for pattern in insensitivity_indicators:
            if re.search(pattern, response_lower):
                sensitivity_score -= 0.2

        return max(0.0, min(1.0, sensitivity_score))

    def _calculate_coherence(self, prediction: str, reference: str) -> float:
        """Calculate coherence of the dialogue response."""
        pred_sentences = [s.strip() for s in prediction.split('.') if s.strip()]

        if len(pred_sentences) < 1:
            return 0.0

        # Check for logical flow
        coherence_score = 0.5

        # Check for discourse markers
        discourse_markers = [
            r"\bfirst\b", r"\bsecond\b", r"\bnext\b", r"\bthen\b", r"\bfinally\b",
            r"\bhowever\b", r"\bmoreover\b", r"\bfurthermore\b", r"\btherefore\b"
        ]

        pred_lower = prediction.lower()
        for pattern in discourse_markers:
            if re.search(pattern, pred_lower):
                coherence_score += 0.1

        return min(1.0, coherence_score)

    def _calculate_informativeness(self,
                                 response: str,
                                 cultural_context: Dict[str, Any]) -> float:
        """Calculate how informative the response is about the culture."""
        culture = cultural_context.get("culture", "")
        response_lower = response.lower()

        informativeness_score = 0.0

        # Check for cultural information
        informative_patterns = [
            r"\bknown for\b", r"\bfamous for\b", r"\bcharacterized by\b",
            r"\btradition.*includes\b", r"\bpractice.*involves\b",
            r"\bbelieve.*that\b", r"\bvalue.*\w+\b"
        ]

        for pattern in informative_patterns:
            if re.search(pattern, response_lower):
                informativeness_score += 0.15

        # Check for specific cultural details
        if culture and culture.lower() in response_lower:
            informativeness_score += 0.2

        # Check for concrete examples
        example_patterns = [
            r"\bfor example\b", r"\bsuch as\b", r"\bincluding\b",
            r"\blike\b.*\w+", r"\bspecifically\b"
        ]

        for pattern in example_patterns:
            if re.search(pattern, response_lower):
                informativeness_score += 0.1

        return min(1.0, informativeness_score)