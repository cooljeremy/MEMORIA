
import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import spacy
from langdetect import detect
import logging

logger = logging.getLogger(__name__)

class SensitivityLevel(Enum):
    """Cultural sensitivity levels"""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    SACRED = "sacred"
    FORBIDDEN = "forbidden"

@dataclass
class CulturalSensitivityResult:
    """Result of cultural sensitivity analysis"""
    sensitivity_level: SensitivityLevel
    sacred_flags: List[str]
    cultural_warnings: List[str]
    restricted_content: List[str]
    confidence_score: float
    recommendations: List[str]

@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    primary_language: str
    confidence: float
    language_family: str
    script_type: str
    cultural_context: Dict[str, Any]


class SacredBoundaryDetector:
    """
    Detector for sacred and restricted cultural content

    Implements protection mechanisms for sacred knowledge, rituals,
    and culturally sensitive information as outlined in the paper.
    """

    def __init__(self, cultural_ontology_path: Optional[str] = None):
        self.sacred_keywords = self._load_sacred_keywords()
        self.cultural_ontology = self._load_cultural_ontology(cultural_ontology_path)
        self.restriction_patterns = self._compile_restriction_patterns()

    def _load_sacred_keywords(self) -> Dict[str, Set[str]]:
        """Load sacred keywords by cultural tradition"""
        # In practice, this would be loaded from cultural expert annotations
        sacred_keywords = {
            "native_american": {
                "sacred_sites", "vision_quest", "sweat_lodge", "medicine_wheel",
                "sacred_pipe", "sundance", "smudging_ceremony", "spiritual_leader",
                "sacred_directions", "ancestor_spirits", "tribal_secrets",
                "healing_ceremonies", "sacred_songs", "ceremonial_objects"
            },
            "tibetan_buddhist": {
                "sacred_texts", "tantric_practices", "initiation_rituals",
                "secret_mantras", "dakini", "terma", "guru_yoga",
                "empowerment_ceremonies", "sacred_dances", "monastery_secrets",
                "meditation_techniques", "spiritual_lineages"
            },
            "aboriginal_australian": {
                "songlines", "dreaming_stories", "sacred_sites", "initiation_secrets",
                "men_only_ceremonies", "women_only_ceremonies", "ancestor_spirits",
                "sacred_objects", "traditional_law", "cultural_protocols",
                "spiritual_knowledge", "clan_secrets"
            },
            "hindu": {
                "sacred_mantras", "temple_rituals", "spiritual_initiation",
                "guru_disciple", "sacred_texts", "tantric_knowledge",
                "meditation_secrets", "pilgrimage_sites", "sacred_geometry",
                "ritual_implements", "spiritual_lineages"
            },
            "generic": {
                "sacred", "secret", "forbidden", "restricted", "private_ceremony",
                "initiation", "spiritual_secret", "holy", "taboo", "clan_knowledge",
                "family_secret", "ancestral_knowledge", "ritual_secret"
            }
        }
        return sacred_keywords

    def _load_cultural_ontology(self, path: Optional[str]) -> Dict[str, Any]:
        """Load cultural knowledge ontology"""
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cultural ontology: {e}")

        # Default ontology structure
        return {
            "access_levels": {
                "public": "Openly shareable cultural knowledge",
                "community": "Requires community permission",
                "restricted": "Limited to cultural members",
                "sacred": "Sacred knowledge with strict protocols",
                "forbidden": "Should not be shared outside community"
            },
            "content_types": {
                "narrative": {"sensitivity": "low", "sharing": "conditional"},
                "ritual": {"sensitivity": "high", "sharing": "restricted"},
                "craft": {"sensitivity": "medium", "sharing": "educational"},
                "sacred_text": {"sensitivity": "very_high", "sharing": "forbidden"},
                "healing": {"sensitivity": "high", "sharing": "practitioner_only"}
            }
        }

    def _compile_restriction_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for restricted content detection"""
        patterns = [
            # Sacred/secret content indicators
            r'\b(?:sacred|secret|forbidden|restricted|private)\s+(?:ceremony|ritual|knowledge|practice)\b',
            r'\b(?:only|exclusively)\s+for\s+(?:initiated|members|practitioners|elders)\b',
            r'\bmust\s+not\s+(?:be\s+)?(?:shared|told|revealed|disclosed)\b',
            r'\b(?:confidential|classified|protected)\s+(?:cultural|traditional|spiritual)\b',
            r'\b(?:male|female)\s+only\s+(?:ceremony|ritual|knowledge|practice)\b',
            r'\brequires?\s+(?:permission|authorization|blessing)\s+from\b',
            r'\b(?:elder|spiritual\s+leader|medicine\s+person)\s+approval\s+(?:required|needed)\b'
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def detect_sacred_content(self, text: str, culture: str) -> CulturalSensitivityResult:
        """
        Detect sacred and restricted content in text

        Args:
            text: Text to analyze
            culture: Cultural context

        Returns:
            CulturalSensitivityResult with sensitivity analysis
        """
        sacred_flags = []
        cultural_warnings = []
        restricted_content = []
        sensitivity_level = SensitivityLevel.PUBLIC

        # Check sacred keywords for the specific culture
        culture_keywords = self.sacred_keywords.get(culture, set())
        generic_keywords = self.sacred_keywords.get("generic", set())
        all_keywords = culture_keywords.union(generic_keywords)

        text_lower = text.lower()
        found_keywords = [kw for kw in all_keywords if kw.replace("_", " ") in text_lower]

        if found_keywords:
            sacred_flags.extend(found_keywords)
            sensitivity_level = SensitivityLevel.RESTRICTED

        # Check restriction patterns
        for pattern in self.restriction_patterns:
            matches = pattern.findall(text)
            if matches:
                restricted_content.extend(matches)
                sensitivity_level = SensitivityLevel.SACRED

        # Cultural-specific analysis
        if culture in ["native_american", "aboriginal_australian"]:
            if any(term in text_lower for term in ["ceremony", "ritual", "spiritual"]):
                cultural_warnings.append("Indigenous cultural content requires community validation")

        if culture in ["tibetan_buddhist", "hindu"]:
            if any(term in text_lower for term in ["meditation", "mantra", "initiation"]):
                cultural_warnings.append("Religious content may require practitioner review")

        # Determine sensitivity level
        if restricted_content:
            sensitivity_level = SensitivityLevel.SACRED
        elif len(sacred_flags) > 2:
            sensitivity_level = SensitivityLevel.RESTRICTED
        elif sacred_flags:
            sensitivity_level = SensitivityLevel.RESTRICTED

        # Calculate confidence score
        confidence_factors = [
            len(found_keywords) * 0.3,
            len(restricted_content) * 0.5,
            len(cultural_warnings) * 0.2
        ]
        confidence_score = min(sum(confidence_factors), 1.0)

        # Generate recommendations
        recommendations = []
        if sensitivity_level in [SensitivityLevel.SACRED, SensitivityLevel.RESTRICTED]:
            recommendations.append("Require cultural community permission before use")
            recommendations.append("Implement access controls for this content")

        if sacred_flags:
            recommendations.append("Review with cultural experts for appropriateness")

        return CulturalSensitivityResult(
            sensitivity_level=sensitivity_level,
            sacred_flags=sacred_flags,
            cultural_warnings=cultural_warnings,
            restricted_content=restricted_content,
            confidence_score=confidence_score,
            recommendations=recommendations
        )

    def apply_protection_filters(self, text: str, sensitivity_result: CulturalSensitivityResult) -> str:
        """Apply protection filters to sensitive content"""

        if sensitivity_result.sensitivity_level == SensitivityLevel.FORBIDDEN:
            return "[CONTENT REMOVED - SACRED/FORBIDDEN]"

        if sensitivity_result.sensitivity_level == SensitivityLevel.SACRED:
            # Redact specific sacred content
            protected_text = text
            for sacred_term in sensitivity_result.sacred_flags:
                protected_text = re.sub(
                    rf'\b{re.escape(sacred_term.replace("_", " "))}\b',
                    '[SACRED CONTENT REDACTED]',
                    protected_text,
                    flags=re.IGNORECASE
                )
            return protected_text

        return text  # Public content unchanged


class LanguageDetector:
    """
    Advanced language detection with cultural context

    Detects language, script, and cultural linguistic context
    for appropriate processing and cultural sensitivity.
    """

    def __init__(self):
        self.language_families = self._build_language_families()
        self.cultural_language_mapping = self._build_cultural_mapping()

    def _build_language_families(self) -> Dict[str, str]:
        """Build language to family mapping"""
        return {
            # Indo-European
            "en": "indo_european", "es": "indo_european", "fr": "indo_european",
            "de": "indo_european", "it": "indo_european", "pt": "indo_european",
            "ru": "indo_european", "hi": "indo_european", "bn": "indo_european",
            "ur": "indo_european", "fa": "indo_european",

            # Sino-Tibetan
            "zh": "sino_tibetan", "zh-cn": "sino_tibetan", "zh-tw": "sino_tibetan",
            "bo": "sino_tibetan", "my": "sino_tibetan",

            # Niger-Congo
            "sw": "niger_congo", "yo": "niger_congo", "ig": "niger_congo",
            "ha": "niger_congo", "zu": "niger_congo",

            # Austronesian
            "id": "austronesian", "ms": "austronesian", "tl": "austronesian",
            "mi": "austronesian", "fj": "austronesian",

            # Afro-Asiatic
            "ar": "afro_asiatic", "he": "afro_asiatic", "am": "afro_asiatic",

            # Others
            "ja": "japonic", "ko": "koreanic", "fi": "uralic",
            "hu": "uralic", "tr": "turkic", "th": "tai_kadai"
        }

    def _build_cultural_mapping(self) -> Dict[str, List[str]]:
        """Build cultural tradition to language mapping"""
        return {
            "chinese_han": ["zh", "zh-cn", "zh-tw"],
            "japanese": ["ja"],
            "korean": ["ko"],
            "tibetan": ["bo", "zh"],
            "hindu": ["hi", "bn", "ur", "en"],
            "islamic": ["ar", "fa", "ur", "id", "ms"],
            "native_american": ["en", "es"],  # Plus many indigenous languages
            "aboriginal_australian": ["en"],  # Plus many indigenous languages
            "african_traditional": ["sw", "yo", "ig", "ha", "zu", "en", "fr"],
            "european_traditional": ["en", "de", "fr", "it", "es", "pt", "ru"],
            "latin_american": ["es", "pt", "en"],
            "pacific_islander": ["en", "mi", "fj", "id"],
            "andalusian": ["es", "en"],
            "yoruba": ["yo", "en"],
            "javanese": ["id", "en"],
            "dogon": ["fr", "en"],
            "mandinka": ["fr", "en"],
            "lakota": ["en"],
            "navajo": ["en"]
        }

    def detect_language(self, text: str, cultural_context: Optional[str] = None) -> LanguageDetectionResult:
        """
        Detect language with cultural context awareness

        Args:
            text: Text to analyze
            cultural_context: Known cultural context

        Returns:
            LanguageDetectionResult with detection details
        """
        try:
            # Primary language detection
            detected_lang = detect(text)
            confidence = 0.8  # Placeholder - would use more sophisticated detection

            # Get language family
            language_family = self.language_families.get(detected_lang, "unknown")

            # Determine script type
            script_type = self._detect_script_type(text)

            # Build cultural context
            cultural_info = {
                "detected_language": detected_lang,
                "language_family": language_family,
                "script_type": script_type
            }

            # Add cultural context if known
            if cultural_context:
                expected_languages = self.cultural_language_mapping.get(cultural_context, [])
                cultural_info["expected_languages"] = expected_languages
                cultural_info["context_match"] = detected_lang in expected_languages

            return LanguageDetectionResult(
                primary_language=detected_lang,
                confidence=confidence,
                language_family=language_family,
                script_type=script_type,
                cultural_context=cultural_info
            )

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return LanguageDetectionResult(
                primary_language="unknown",
                confidence=0.0,
                language_family="unknown",
                script_type="unknown",
                cultural_context={}
            )

    def _detect_script_type(self, text: str) -> str:
        """Detect script/writing system type"""
        if re.search(r'[\u4e00-\u9fff]', text):
            return "chinese_characters"
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "japanese_hiragana_katakana"
        elif re.search(r'[\u0600-\u06ff]', text):
            return "arabic_script"
        elif re.search(r'[\u0900-\u097f]', text):
            return "devanagari"
        elif re.search(r'[\u0f00-\u0fff]', text):
            return "tibetan_script"
        elif re.search(r'[a-zA-Z]', text):
            return "latin_script"
        else:
            return "mixed_or_unknown"


class CulturalPreprocessor:
    """
    Cultural-aware text preprocessing

    Handles cultural context, language normalization,
    and preparation for cultural AI models.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sacred_detector = SacredBoundaryDetector()
        self.language_detector = LanguageDetector()

        # Load spaCy models if available
        self.nlp_models = {}
        try:
            self.nlp_models["en"] = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not available")

    def preprocess_cultural_text(
        self,
        text: str,
        culture: str,
        preserve_cultural_markers: bool = True
    ) -> Dict[str, Any]:
        """
        Preprocess text with cultural awareness

        Args:
            text: Input text
            culture: Cultural context
            preserve_cultural_markers: Whether to preserve cultural-specific elements

        Returns:
            Dictionary with preprocessed text and metadata
        """

        # Language detection
        lang_result = self.language_detector.detect_language(text, culture)

        # Sacred content detection
        sensitivity_result = self.sacred_detector.detect_sacred_content(text, culture)

        # Apply protection if needed
        if sensitivity_result.sensitivity_level in [SensitivityLevel.SACRED, SensitivityLevel.FORBIDDEN]:
            protected_text = self.sacred_detector.apply_protection_filters(text, sensitivity_result)
        else:
            protected_text = text

        # Cultural text normalization
        normalized_text = self._normalize_cultural_text(
            protected_text,
            culture,
            lang_result.primary_language,
            preserve_cultural_markers
        )

        # Extract cultural entities
        cultural_entities = self._extract_cultural_entities(normalized_text, culture)

        return {
            "original_text": text,
            "processed_text": normalized_text,
            "language_info": lang_result,
            "sensitivity_info": sensitivity_result,
            "cultural_entities": cultural_entities,
            "processing_metadata": {
                "culture": culture,
                "preserve_markers": preserve_cultural_markers,
                "protection_applied": sensitivity_result.sensitivity_level != SensitivityLevel.PUBLIC
            }
        }

    def _normalize_cultural_text(
        self,
        text: str,
        culture: str,
        language: str,
        preserve_markers: bool
    ) -> str:
        """Normalize text while preserving cultural elements"""

        normalized = text

        if preserve_markers:
            # Preserve cultural terms and names
            cultural_terms = self._identify_cultural_terms(text, culture)

            # Basic normalization
            normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
            normalized = normalized.strip()

            # Preserve special cultural punctuation and formatting
            if culture in ["chinese_han", "japanese"]:
                # Preserve Chinese/Japanese punctuation
                pass
            elif culture in ["arabic", "islamic"]:
                # Preserve Arabic script features
                pass

        else:
            # Standard normalization
            normalized = re.sub(r'\s+', ' ', text).strip()

        return normalized

    def _identify_cultural_terms(self, text: str, culture: str) -> List[str]:
        """Identify cultural-specific terms to preserve"""
        cultural_terms = []

        # Culture-specific term patterns
        term_patterns = {
            "japanese": [r'\b\w+(?:san|sama|chan|kun)\b', r'\b[a-z]+(?:do|jutsu|kata)\b'],
            "chinese_han": [r'\b\w+(?:shi|fu|laoshi)\b'],
            "hindu": [r'\b\w+(?:ji|maharaj|swami)\b'],
            "islamic": [r'\b\w+(?:imam|sheikh|hafiz)\b'],
            "native_american": [r'\b\w+(?:chief|elder|medicine\s+\w+)\b']
        }

        patterns = term_patterns.get(culture, [])
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            cultural_terms.extend(matches)

        return cultural_terms

    def _extract_cultural_entities(self, text: str, culture: str) -> List[Dict[str, Any]]:
        """Extract cultural entities from text"""
        entities = []

        # Use spaCy if available
        if "en" in self.nlp_models and text:
            doc = self.nlp_models["en"](text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "cultural_relevance": self._assess_cultural_relevance(ent.text, culture)
                })

        return entities

    def _assess_cultural_relevance(self, entity_text: str, culture: str) -> float:
        """Assess cultural relevance of an entity"""
        # Placeholder scoring - would use cultural knowledge bases
        cultural_keywords = {
            "japanese": ["tea", "ceremony", "samurai", "zen", "shrine", "temple"],
            "chinese_han": ["dragon", "ancestor", "harmony", "temple", "emperor"],
            "hindu": ["karma", "dharma", "temple", "guru", "yoga", "mantra"],
            "native_american": ["spirit", "earth", "ceremony", "sacred", "tribe", "elder"]
        }

        keywords = cultural_keywords.get(culture, [])
        entity_lower = entity_text.lower()

        relevance = sum(1 for kw in keywords if kw in entity_lower)
        return min(relevance / max(len(keywords), 1), 1.0)


class QualityValidator:
    """
    Quality validation for cultural heritage data

    Implements quality checks for cultural accuracy,
    completeness, and authenticity as described in the paper.
    """

    def __init__(self, quality_thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = quality_thresholds or {
            "cultural_accuracy": 0.8,
            "completeness": 0.7,
            "authenticity": 0.85,
            "source_reliability": 0.75,
            "overall_quality": 0.8
        }

    def validate_cultural_sample(
        self,
        sample_text: str,
        culture: str,
        source_metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Validate quality of a cultural heritage sample

        Args:
            sample_text: Text content to validate
            culture: Cultural context
            source_metadata: Metadata about source and collection

        Returns:
            Dictionary of quality scores
        """

        scores = {}

        # Cultural accuracy assessment
        scores["cultural_accuracy"] = self._assess_cultural_accuracy(sample_text, culture)

        # Completeness assessment
        scores["completeness"] = self._assess_completeness(sample_text, source_metadata)

        # Authenticity assessment
        scores["authenticity"] = self._assess_authenticity(sample_text, culture, source_metadata)

        # Source reliability
        scores["source_reliability"] = self._assess_source_reliability(source_metadata)

        # Language quality
        scores["language_quality"] = self._assess_language_quality(sample_text)

        # Overall quality (weighted average)
        weights = {
            "cultural_accuracy": 0.3,
            "completeness": 0.2,
            "authenticity": 0.25,
            "source_reliability": 0.15,
            "language_quality": 0.1
        }

        scores["overall_quality"] = sum(
            scores[metric] * weight for metric, weight in weights.items()
        )

        return scores

    def _assess_cultural_accuracy(self, text: str, culture: str) -> float:
        """Assess cultural accuracy of content"""
        # Placeholder implementation - would use cultural knowledge bases

        # Check for cultural inconsistencies
        accuracy_indicators = {
            "japanese": ["honor", "respect", "harmony", "tradition", "ceremony"],
            "chinese_han": ["harmony", "balance", "ancestor", "tradition", "wisdom"],
            "hindu": ["spiritual", "divine", "sacred", "tradition", "wisdom"],
            "native_american": ["sacred", "spirit", "earth", "ancestor", "tradition"],
            "islamic": ["sacred", "blessed", "tradition", "community", "wisdom"]
        }

        indicators = accuracy_indicators.get(culture, [])
        text_lower = text.lower()

        matching_indicators = sum(1 for indicator in indicators if indicator in text_lower)
        accuracy_score = min(matching_indicators / max(len(indicators), 1), 1.0)

        # Penalize cultural mismatches
        if culture == "japanese" and "chinese" in text_lower:
            accuracy_score *= 0.8
        elif culture == "hindu" and "buddhist" in text_lower:
            accuracy_score *= 0.9

        return max(accuracy_score, 0.5)  # Minimum baseline

    def _assess_completeness(self, text: str, metadata: Dict[str, Any]) -> float:
        """Assess completeness of cultural information"""

        completeness_factors = []

        # Text length factor
        text_length = len(text.split())
        if text_length > 100:
            completeness_factors.append(1.0)
        elif text_length > 50:
            completeness_factors.append(0.8)
        else:
            completeness_factors.append(0.5)

        # Metadata completeness
        required_fields = ["source", "cultural_context", "collection_date"]
        present_fields = sum(1 for field in required_fields if field in metadata)
        metadata_completeness = present_fields / len(required_fields)
        completeness_factors.append(metadata_completeness)

        # Content depth (presence of cultural details)
        depth_indicators = ["description", "significance", "practice", "tradition", "meaning"]
        depth_score = sum(1 for indicator in depth_indicators if indicator in text.lower())
        depth_completeness = min(depth_score / len(depth_indicators), 1.0)
        completeness_factors.append(depth_completeness)

        return np.mean(completeness_factors)

    def _assess_authenticity(self, text: str, culture: str, metadata: Dict[str, Any]) -> float:
        """Assess authenticity of cultural content"""

        authenticity_score = 0.5  # Base score

        # Source type scoring
        source_type = metadata.get("source_type", "unknown")
        source_scores = {
            "cultural_expert": 1.0,
            "community_elder": 0.95,
            "academic_research": 0.85,
            "cultural_institution": 0.9,
            "traditional_practitioner": 0.92,
            "unesco_documentation": 0.98,
            "oral_tradition": 0.88,
            "unknown": 0.5
        }
        authenticity_score *= source_scores.get(source_type, 0.5)

        # Cultural context consistency
        if metadata.get("cultural_validation"):
            authenticity_score *= 1.1

        if metadata.get("community_approval"):
            authenticity_score *= 1.15

        # First-hand vs. second-hand information
        if "I witnessed" in text or "I participated" in text:
            authenticity_score *= 1.1
        elif "According to" in text or "It is said" in text:
            authenticity_score *= 0.9

        return min(authenticity_score, 1.0)

    def _assess_source_reliability(self, metadata: Dict[str, Any]) -> float:
        """Assess reliability of the data source"""

        reliability_factors = []

        # Source authority
        authority_indicators = ["unesco", "university", "museum", "cultural_center", "traditional_authority"]
        source = metadata.get("source", "").lower()
        authority_score = sum(0.2 for indicator in authority_indicators if indicator in source)
        reliability_factors.append(min(authority_score, 1.0))

        # Publication/documentation quality
        if metadata.get("peer_reviewed"):
            reliability_factors.append(1.0)
        elif metadata.get("expert_validated"):
            reliability_factors.append(0.9)
        else:
            reliability_factors.append(0.6)

        # Date and relevance
        collection_year = metadata.get("collection_year", 2020)
        if collection_year >= 2020:
            reliability_factors.append(1.0)
        elif collection_year >= 2010:
            reliability_factors.append(0.9)
        else:
            reliability_factors.append(0.7)

        return np.mean(reliability_factors)

    def _assess_language_quality(self, text: str) -> float:
        """Assess language quality and coherence"""

        quality_factors = []

        # Basic language metrics
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])

        if 10 <= avg_sentence_length <= 25:
            quality_factors.append(1.0)
        elif 5 <= avg_sentence_length <= 35:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.6)

        # Vocabulary richness (unique words / total words)
        words = text.lower().split()
        if words:
            vocabulary_richness = len(set(words)) / len(words)
            quality_factors.append(min(vocabulary_richness * 2, 1.0))
        else:
            quality_factors.append(0.0)

        # Grammar and coherence (simplified check)
        grammar_score = 1.0
        if text.count('.') == 0 and len(text.split()) > 20:
            grammar_score *= 0.7  # Missing punctuation
        if re.search(r'\b\w+\b.*\b\1\b', text):  # Repeated words
            grammar_score *= 0.9

        quality_factors.append(grammar_score)

        return np.mean(quality_factors)

    def meets_quality_threshold(self, quality_scores: Dict[str, float]) -> bool:
        """Check if sample meets quality thresholds"""

        for metric, threshold in self.thresholds.items():
            if quality_scores.get(metric, 0) < threshold:
                return False

        return True