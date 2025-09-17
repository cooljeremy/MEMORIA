import re
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CulturalScore:
    """5-dimensional Cultural Score (CS) metric."""
    structural_authenticity: float  # SA: 0-1
    motif_fidelity: float  # MF: 0-1
    linguistic_authenticity: float  # LA: 0-1
    value_alignment: float  # VA: 0-1
    transmission_appropriateness: float  # TA: 0-1
    overall_score: float = field(init=False)
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "structural_authenticity": 0.25,
        "motif_fidelity": 0.20,
        "linguistic_authenticity": 0.20,
        "value_alignment": 0.20,
        "transmission_appropriateness": 0.15
    })

    def __post_init__(self):
        """Calculate overall score as weighted average."""
        self.overall_score = (
            self.structural_authenticity * self.dimension_weights["structural_authenticity"] +
            self.motif_fidelity * self.dimension_weights["motif_fidelity"] +
            self.linguistic_authenticity * self.dimension_weights["linguistic_authenticity"] +
            self.value_alignment * self.dimension_weights["value_alignment"] +
            self.transmission_appropriateness * self.dimension_weights["transmission_appropriateness"]
        )


@dataclass
class CulturalFidelity:
    """4-dimensional Cultural Fidelity (CF) metric for translation tasks."""
    conceptual_preservation: float  # CP: 0-1
    metaphorical_mapping: float  # MM: 0-1
    pragmatic_accuracy: float  # PA: 0-1
    sacred_knowledge_handling: float  # SKH: 0-1
    overall_score: float = field(init=False)
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "conceptual_preservation": 0.30,
        "metaphorical_mapping": 0.25,
        "pragmatic_accuracy": 0.25,
        "sacred_knowledge_handling": 0.20
    })

    def __post_init__(self):
        """Calculate overall score as weighted average."""
        self.overall_score = (
            self.conceptual_preservation * self.dimension_weights["conceptual_preservation"] +
            self.metaphorical_mapping * self.dimension_weights["metaphorical_mapping"] +
            self.pragmatic_accuracy * self.dimension_weights["pragmatic_accuracy"] +
            self.sacred_knowledge_handling * self.dimension_weights["sacred_knowledge_handling"]
        )


class CulturalKnowledgeBase:
    """Knowledge base for cultural concepts, terminology, and sensitivity rules."""

    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = Path(kb_path) if kb_path else Path(__file__).parent / "cultural_kb"
        self.cultural_concepts = {}
        self.cultural_terminology = {}
        self.sacred_boundaries = {}
        self.domain_specific_rules = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load cultural knowledge base from files."""
        kb_path = self.kb_path
        kb_path.mkdir(parents=True, exist_ok=True)

        # Load or create cultural concepts
        concepts_file = kb_path / "cultural_concepts.json"
        if concepts_file.exists():
            with open(concepts_file, 'r', encoding='utf-8') as f:
                self.cultural_concepts = json.load(f)
        else:
            self._create_sample_knowledge_base()

    def _create_sample_knowledge_base(self):
        """Create sample cultural knowledge base."""
        logger.info("Creating sample cultural knowledge base...")

        # Sample cultural concepts
        self.cultural_concepts = {
            "chinese": {
                "concepts": ["filial_piety", "harmony", "respect_for_elders", "tea_ceremony", "calligraphy"],
                "terminology": ["孝顺", "和谐", "茶道", "书法", "传统"],
                "sacred_boundaries": ["ancestral_worship", "religious_practices"],
                "domains": {
                    "oral_traditions": ["folk_tales", "proverbs", "legends"],
                    "performing_arts": ["opera", "martial_arts", "traditional_music"],
                    "social_practices": ["festivals", "ceremonies", "customs"],
                    "traditional_craftsmanship": ["pottery", "textiles", "woodworking"],
                    "knowledge_practices": ["traditional_medicine", "astronomy", "agriculture"]
                }
            },
            "japanese": {
                "concepts": ["wa", "ikigai", "omotenashi", "mono_no_aware", "bushido"],
                "terminology": ["和", "生きがい", "おもてなし", "物の哀れ", "武士道"],
                "sacred_boundaries": ["shrine_practices", "ancestral_reverence"],
                "domains": {
                    "oral_traditions": ["folklore", "poetry", "storytelling"],
                    "performing_arts": ["noh", "kabuki", "traditional_dance"],
                    "social_practices": ["tea_ceremony", "seasonal_festivals"],
                    "traditional_craftsmanship": ["pottery", "metalwork", "textiles"],
                    "knowledge_practices": ["traditional_medicine", "martial_arts"]
                }
            },
            "tibetan": {
                "concepts": ["compassion", "impermanence", "karma", "dharma", "mindfulness"],
                "terminology": ["བླ་མ་", "དཀར་པོ་", "ཆོས་", "སྒྲུབ་པ་"],
                "sacred_boundaries": ["religious_texts", "monastery_practices", "sacred_sites"],
                "domains": {
                    "oral_traditions": ["prayers", "mantras", "legends"],
                    "performing_arts": ["ritual_dances", "chanting", "music"],
                    "social_practices": ["pilgrimages", "festivals", "ceremonies"],
                    "traditional_craftsmanship": ["thangka_painting", "metalwork", "textiles"],
                    "knowledge_practices": ["traditional_medicine", "astrology", "philosophy"]
                }
            }
        }

        # Sacred boundary rules
        self.sacred_boundaries = {
            "general_rules": [
                "avoid_commercializing_sacred_practices",
                "respect_religious_prohibitions",
                "handle_ancestral_content_with_reverence",
                "avoid_misrepresenting_rituals"
            ],
            "culture_specific": {
                "tibetan": ["no_casual_use_of_mantras", "respect_for_dalai_lama", "sacred_site_reverence"],
                "aboriginal_australian": ["dreamtime_story_restrictions", "sacred_site_protection"],
                "native_american": ["ceremonial_object_respect", "spiritual_practice_boundaries"]
            }
        }

        # Save knowledge base
        with open(self.kb_path / "cultural_concepts.json", 'w', encoding='utf-8') as f:
            json.dump(self.cultural_concepts, f, ensure_ascii=False, indent=2)

        with open(self.kb_path / "sacred_boundaries.json", 'w', encoding='utf-8') as f:
            json.dump(self.sacred_boundaries, f, ensure_ascii=False, indent=2)

    def get_cultural_concepts(self, culture: str) -> Dict[str, Any]:
        """Get cultural concepts for a specific culture."""
        return self.cultural_concepts.get(culture, {})

    def get_sacred_boundaries(self, culture: str) -> List[str]:
        """Get sacred boundary rules for a specific culture."""
        general_rules = self.sacred_boundaries.get("general_rules", [])
        specific_rules = self.sacred_boundaries.get("culture_specific", {}).get(culture, [])
        return general_rules + specific_rules

    def is_culturally_appropriate_term(self, term: str, culture: str, context: str) -> bool:
        """Check if a term is culturally appropriate in the given context."""
        cultural_data = self.get_cultural_concepts(culture)
        terminology = cultural_data.get("terminology", [])

        # Simple heuristic: check if term appears in cultural terminology
        # In a real implementation, this would be more sophisticated
        return any(term in t or t in term for t in terminology)

    def contains_sacred_content(self, text: str, culture: str) -> Tuple[bool, List[str]]:
        """Check if text contains sacred/sensitive content."""
        sacred_rules = self.get_sacred_boundaries(culture)
        violations = []

        # Simple pattern matching for demonstration
        # In practice, this would use more sophisticated NLP techniques
        text_lower = text.lower()

        for rule in sacred_rules:
            if any(keyword in text_lower for keyword in rule.split('_')):
                violations.append(rule)

        return len(violations) > 0, violations


class CulturalMetricsCalculator:
    """Calculator for Cultural Score and Cultural Fidelity metrics."""

    def __init__(self, knowledge_base: Optional[CulturalKnowledgeBase] = None):
        self.kb = knowledge_base or CulturalKnowledgeBase()

    def calculate_cultural_score(self,
                                text: str,
                                culture: str,
                                expected_cultural_elements: Dict[str, Any],
                                domain: str,
                                reference_text: Optional[str] = None) -> CulturalScore:
        """
        Calculate 5-dimensional Cultural Score.

        Args:
            text: Generated text to evaluate
            culture: Target culture
            expected_cultural_elements: Expected cultural elements
            domain: UNESCO ICH domain
            reference_text: Reference text for comparison 

        Returns:
            CulturalScore with all dimensions
        """
        # Get cultural context
        cultural_data = self.kb.get_cultural_concepts(culture)

        # Calculate each dimension according to LaTeX paper definitions
        sa_score = self._calculate_structural_authenticity(text, culture, cultural_data, domain)
        mf_score = self._calculate_motif_fidelity(text, culture, cultural_data, expected_cultural_elements)
        la_score = self._calculate_linguistic_authenticity(text, culture, cultural_data)
        va_score = self._calculate_value_alignment(text, culture, cultural_data)
        ta_score = self._calculate_transmission_appropriateness(text, expected_cultural_elements, reference_text)

        return CulturalScore(
            structural_authenticity=sa_score,
            motif_fidelity=mf_score,
            linguistic_authenticity=la_score,
            value_alignment=va_score,
            transmission_appropriateness=ta_score
        )

    def calculate_cultural_fidelity(self,
                                  source_text: str,
                                  translated_text: str,
                                  source_culture: str,
                                  target_culture: str) -> CulturalFidelity:
        """
        Calculate 4-dimensional Cultural Fidelity for translation tasks.

        Args:
            source_text: Original text
            translated_text: Translated text
            source_culture: Source culture
            target_culture: Target culture

        Returns:
            CulturalFidelity with all dimensions
        """
        cp_score = self._calculate_conceptual_preservation(source_text, translated_text, source_culture)
        mm_score = self._calculate_metaphorical_mapping(source_text, translated_text, source_culture, target_culture)
        pa_score = self._calculate_pragmatic_accuracy(source_text, translated_text, source_culture, target_culture)
        skh_score = self._calculate_sacred_knowledge_handling(translated_text, source_culture, target_culture)

        return CulturalFidelity(
            conceptual_preservation=cp_score,
            metaphorical_mapping=mm_score,
            pragmatic_accuracy=pa_score,
            sacred_knowledge_handling=skh_score
        )

    def _calculate_structural_authenticity(self,
                                      text: str,
                                      culture: str,
                                      cultural_data: Dict[str, Any],
                                      domain: str) -> float:
        """
        Calculate Structural Authenticity (SA) dimension.

        Formula: f_SA(x, C) = α·Align(x, T_c) + β·Struct(x, P_c) + γ·Form(x, F_c)
        where α=0.4, β=0.35, γ=0.25
        """
        if not cultural_data:
            return 0.5

        # Component weights (α, β, γ)
        alpha, beta, gamma = 0.4, 0.35, 0.25

        # 1. Align(x, T_c): Sequence alignment with cultural templates
        align_score = self._compute_template_alignment(text, culture, cultural_data)

        # 2. Struct(x, P_c): Narrative structure analysis
        struct_score = self._compute_narrative_structure(text, culture, domain)

        # 3. Form(x, F_c): Formulaic expression density
        form_score = self._compute_formulaic_density(text, cultural_data)

        # Weighted combination
        sa_score = alpha * align_score + beta * struct_score + gamma * form_score
        return min(1.0, max(0.0, sa_score))

    def _compute_template_alignment(self, text: str, culture: str, cultural_data: Dict[str, Any]) -> float:
        """Compute sequence alignment with cultural templates using modified Needleman-Wunsch."""
        # Get cultural narrative templates
        templates = cultural_data.get("narrative_templates", [])
        if not templates:
            return 0.5

        text_tokens = text.lower().split()
        max_alignment = 0.0

        for template in templates[:5]:  # Limit to top 5 templates for efficiency
            template_tokens = template.lower().split() if isinstance(template, str) else []
            if not template_tokens:
                continue

            # Simplified alignment score based on longest common subsequence
            alignment_score = self._lcs_alignment(text_tokens, template_tokens)
            max_alignment = max(max_alignment, alignment_score)

        return max_alignment

    def _lcs_alignment(self, seq1: List[str], seq2: List[str]) -> float:
        """Compute longest common subsequence alignment score."""
        if not seq1 or not seq2:
            return 0.0

        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Normalize by average length
        lcs_length = dp[m][n]
        avg_length = (m + n) / 2
        return lcs_length / avg_length if avg_length > 0 else 0.0

    def _compute_narrative_structure(self, text: str, culture: str, domain: str) -> float:
        """Analyze narrative structure patterns using graph-based analysis."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 3:
            return 0.3  # Too short for structure analysis

        # Look for cultural narrative patterns
        structure_indicators = {
            "opening": ["once upon", "long ago", "in ancient times", "there was", "there lived"],
            "development": ["then", "next", "after that", "meanwhile", "suddenly"],
            "climax": ["finally", "at last", "in the end", "ultimately"],
            "closing": ["and so", "thus", "therefore", "ever since", "to this day"]
        }

        structure_score = 0.0
        text_lower = text.lower()

        # Check for each structural component
        for component, indicators in structure_indicators.items():
            component_found = any(indicator in text_lower for indicator in indicators)
            if component_found:
                structure_score += 0.25

        # Bonus for culture-specific structures
        if culture in ["chinese", "japanese", "korean"]:  # kishōtenketsu pattern
            if self._detect_kishotenketsu_pattern(sentences):
                structure_score += 0.2

        return min(1.0, structure_score)

    def _detect_kishotenketsu_pattern(self, sentences: List[str]) -> bool:
        """Detect four-act kishōtenketsu narrative pattern."""
        if len(sentences) < 4:
            return False

        # Simplified detection based on sentence content patterns
        # In practice, this would use more sophisticated NLP
        ki_patterns = ["introduction", "setting", "character", "beginning"]
        sho_patterns = ["development", "progress", "continuing", "then"]
        ten_patterns = ["twist", "change", "suddenly", "unexpected", "however"]
        ketsu_patterns = ["conclusion", "result", "finally", "ending", "thus"]

        sections = [sentences[:len(sentences)//4],
                   sentences[len(sentences)//4:len(sentences)//2],
                   sentences[len(sentences)//2:3*len(sentences)//4],
                   sentences[3*len(sentences)//4:]]

        pattern_found = 0
        for i, (section, patterns) in enumerate(zip(sections, [ki_patterns, sho_patterns, ten_patterns, ketsu_patterns])):
            section_text = " ".join(section).lower()
            if any(pattern in section_text for pattern in patterns):
                pattern_found += 1

        return pattern_found >= 3

    def _compute_formulaic_density(self, text: str, cultural_data: Dict[str, Any]) -> float:
        """Assess formulaic expression density using n-gram matching."""
        formulaic_expressions = cultural_data.get("formulaic_expressions", [])
        if not formulaic_expressions:
            return 0.5

        text_lower = text.lower()
        expression_matches = 0

        for expression in formulaic_expressions:
            expr_lower = expression.lower() if isinstance(expression, str) else ""
            if expr_lower and expr_lower in text_lower:
                expression_matches += 1

        # Calculate density
        total_expressions = len(formulaic_expressions)
        if total_expressions == 0:
            return 0.5

        density = expression_matches / total_expressions
        return min(1.0, density * 2.0)  # Scale up to reward presence

    def _calculate_motif_fidelity(self,
                                 text: str,
                                 culture: str,
                                 cultural_data: Dict[str, Any],
                                 expected_elements: Dict[str, Any]) -> float:
        """
        Calculate Motif Fidelity (MF) dimension.

        Formula: f_MF(x, O_c) = (1/|M|) * Σ sim(e_m, O_c) * freq(m, x) * context(m, x)
        """
        if not cultural_data:
            return 0.5

        # Extract motifs from text using ontology-guided detection
        detected_motifs = self._extract_cultural_motifs(text, cultural_data)

        if not detected_motifs:
            return 0.1  # Low score if no motifs detected

        total_score = 0.0

        for motif in detected_motifs:
            # sim(e_m, O_c): Semantic similarity with cultural ontology
            similarity_score = self._compute_motif_similarity(motif, cultural_data)

            # freq(m, x): Motif frequency normalized by text length
            frequency_score = self._compute_motif_frequency(motif, text)

            # context(m, x): Contextual appropriateness
            context_score = self._compute_motif_context(motif, text, cultural_data)

            # Combined score for this motif
            motif_score = similarity_score * frequency_score * context_score
            total_score += motif_score

        # Average across all detected motifs
        return total_score / len(detected_motifs)

    def _extract_cultural_motifs(self, text: str, cultural_data: Dict[str, Any]) -> List[str]:
        """Extract cultural motifs using ontology-guided NER."""
        text_lower = text.lower()
        motifs = []

        # Get cultural concepts that could be motifs
        concepts = cultural_data.get("concepts", [])
        terminology = cultural_data.get("terminology", [])

        # Check for archetypal entities
        archetypal_entities = [
            "hero", "villain", "wise_elder", "trickster", "guardian",
            "ancestor", "spirit", "deity", "monster", "messenger"
        ]

        # Combine all potential motifs
        all_motifs = concepts + terminology + archetypal_entities

        for motif in all_motifs:
            motif_variants = [
                motif,
                motif.replace('_', ' '),
                motif.replace('_', ''),
                motif.capitalize()
            ]

            for variant in motif_variants:
                if variant.lower() in text_lower:
                    motifs.append(motif)
                    break

        return list(set(motifs))  # Remove duplicates

    def _compute_motif_similarity(self, motif: str, cultural_data: Dict[str, Any]) -> float:
        """Compute semantic similarity with cultural ontology using embeddings."""
        # Simplified similarity based on cultural concept matching
        concepts = cultural_data.get("concepts", [])
        domains = cultural_data.get("domains", {})

        # Check if motif is a core cultural concept
        if motif in concepts:
            return 1.0

        # Check similarity with domain concepts
        for domain, domain_concepts in domains.items():
            if motif in domain_concepts:
                return 0.8

        # Check for partial matches (simplified embedding similarity)
        motif_words = set(motif.replace('_', ' ').lower().split())
        max_similarity = 0.0

        for concept in concepts:
            concept_words = set(concept.replace('_', ' ').lower().split())
            if motif_words.intersection(concept_words):
                # Jaccard similarity as proxy for semantic similarity
                intersection = len(motif_words.intersection(concept_words))
                union = len(motif_words.union(concept_words))
                similarity = intersection / union if union > 0 else 0.0
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _compute_motif_frequency(self, motif: str, text: str) -> float:
        """Quantify motif frequency normalized by text length."""
        text_lower = text.lower()
        motif_variants = [
            motif.lower(),
            motif.replace('_', ' ').lower(),
            motif.replace('_', '').lower()
        ]

        total_count = 0
        for variant in motif_variants:
            total_count += text_lower.count(variant)

        # Normalize by text length (in words)
        text_length = len(text.split())
        if text_length == 0:
            return 0.0

        # Frequency score: higher frequency gets higher score (up to a limit)
        frequency = total_count / text_length
        return min(1.0, frequency * 50)  # Scale to reasonable range

    def _compute_motif_context(self, motif: str, text: str, cultural_data: Dict[str, Any]) -> float:
        """Evaluate contextual appropriateness using cultural context classifiers."""
        # Find contexts where motif appears
        sentences = text.split('.')
        motif_contexts = []

        motif_lower = motif.lower().replace('_', ' ')

        for sentence in sentences:
            if motif_lower in sentence.lower():
                motif_contexts.append(sentence.strip())

        if not motif_contexts:
            return 0.0

        # Analyze context appropriateness
        appropriate_contexts = 0

        # Cultural context indicators
        positive_indicators = [
            "traditional", "ancient", "sacred", "ceremonial", "ritual",
            "cultural", "heritage", "ancestor", "spiritual", "reverent"
        ]

        negative_indicators = [
            "modern", "contemporary", "fake", "artificial", "inappropriate",
            "commercialized", "tourist", "stereotypical", "misrepresented"
        ]

        for context in motif_contexts:
            context_lower = context.lower()

            positive_score = sum(1 for indicator in positive_indicators if indicator in context_lower)
            negative_score = sum(1 for indicator in negative_indicators if indicator in context_lower)

            # Context is appropriate if it has positive indicators and few negative ones
            if positive_score > negative_score:
                appropriate_contexts += 1

        return appropriate_contexts / len(motif_contexts) if motif_contexts else 0.0

    def _calculate_linguistic_authenticity(self,
                                          text: str,
                                          culture: str,
                                          cultural_data: Dict[str, Any]) -> float:
        """
        Calculate Linguistic Authenticity (LA) dimension.

        Formula: f_LA(x, L_c) = λ₁·Pros(x, R_c) + λ₂·Style(x, S_c) + λ₃·Dist(x, D_c)
        where λ₁=0.3, λ₂=0.4, λ₃=0.3
        """
        if not cultural_data:
            return 0.5

        # Component weights (λ₁, λ₂, λ₃)
        lambda1, lambda2, lambda3 = 0.3, 0.4, 0.3

        # 1. Pros(x, R_c): Prosodic patterns against cultural rhythm templates
        prosody_score = self._compute_prosodic_patterns(text, culture, cultural_data)

        # 2. Style(x, S_c): Stylistic similarity using KL divergence
        style_score = self._compute_stylistic_similarity(text, culture, cultural_data)

        # 3. Dist(x, D_c): Distributional semantic alignment
        distributional_score = self._compute_distributional_alignment(text, culture, cultural_data)

        # Weighted combination
        la_score = lambda1 * prosody_score + lambda2 * style_score + lambda3 * distributional_score
        return min(1.0, max(0.0, la_score))

    def _compute_prosodic_patterns(self, text: str, culture: str, cultural_data: Dict[str, Any]) -> float:
        """Compute prosodic patterns using rhythm similarity."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0

        total_rhythm_score = 0.0

        for sentence in sentences:
            # Calculate rhythm metrics
            rhythm_score = self._analyze_sentence_rhythm(sentence, culture)
            total_rhythm_score += rhythm_score

        # Average rhythm similarity across all sentences
        return total_rhythm_score / len(sentences)

    def _analyze_sentence_rhythm(self, sentence: str, culture: str) -> float:
        """Analyze rhythm patterns in a sentence."""
        words = sentence.split()
        if len(words) < 3:
            return 0.5

        # Analyze syllable patterns (simplified)
        syllable_counts = []
        for word in words:
            # Simplified syllable counting based on vowel groups
            syllables = max(1, len(re.findall(r'[aeiouAEIOU]+', word)))
            syllable_counts.append(syllables)

        if not syllable_counts:
            return 0.0

        # Calculate rhythm regularity
        avg_syllables = sum(syllable_counts) / len(syllable_counts)
        variance = sum((s - avg_syllables) ** 2 for s in syllable_counts) / len(syllable_counts)

        # Cultural rhythm preferences (simplified heuristics)
        if culture in ["chinese", "japanese", "korean"]:
            # Preference for more regular rhythm patterns
            regularity_score = 1.0 / (1.0 + variance)
        else:
            # More flexible rhythm patterns
            regularity_score = min(1.0, 0.5 + 0.5 / (1.0 + variance))

        return regularity_score

    def _compute_stylistic_similarity(self, text: str, culture: str, cultural_data: Dict[str, Any]) -> float:
        """Compute stylistic similarity using KL divergence between n-gram distributions."""
        # Extract n-grams from text
        text_ngrams = self._extract_ngrams(text, n=2)  # Using bigrams

        # Get cultural style patterns
        cultural_styles = cultural_data.get("stylistic_patterns", [])
        if not cultural_styles:
            return 0.5

        # Calculate n-gram distribution for input text
        text_distribution = self._compute_ngram_distribution(text_ngrams)

        # Calculate similarity with cultural style distributions
        max_similarity = 0.0

        for style_pattern in cultural_styles:
            if isinstance(style_pattern, str):
                style_ngrams = self._extract_ngrams(style_pattern, n=2)
                style_distribution = self._compute_ngram_distribution(style_ngrams)

                # Compute inverse KL divergence (higher = more similar)
                kl_div = self._compute_kl_divergence(text_distribution, style_distribution)
                similarity = 1.0 / (1.0 + kl_div)  # KL(P||Q)^(-1)
                max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _extract_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """Extract n-grams from text."""
        words = text.lower().split()
        if len(words) < n:
            return []

        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))

        return ngrams

    def _compute_ngram_distribution(self, ngrams: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], float]:
        """Compute probability distribution of n-grams."""
        if not ngrams:
            return {}

        from collections import Counter
        ngram_counts = Counter(ngrams)
        total_count = len(ngrams)

        distribution = {}
        for ngram, count in ngram_counts.items():
            distribution[ngram] = count / total_count

        return distribution

    def _compute_kl_divergence(self, p_dist: Dict, q_dist: Dict) -> float:
        """Compute KL divergence between two distributions."""
        if not p_dist or not q_dist:
            return float('inf')

        # Add smoothing for unseen n-grams
        epsilon = 1e-10
        kl_div = 0.0

        all_ngrams = set(p_dist.keys()) | set(q_dist.keys())

        for ngram in all_ngrams:
            p_prob = p_dist.get(ngram, epsilon)
            q_prob = q_dist.get(ngram, epsilon)

            if p_prob > 0:
                kl_div += p_prob * np.log(p_prob / q_prob)

        return kl_div

    def _compute_distributional_alignment(self, text: str, culture: str, cultural_data: Dict[str, Any]) -> float:
        """Compute distributional semantic alignment using cosine similarity."""
        # Simplified implementation using word overlap as proxy for BERT embeddings
        text_words = set(text.lower().split())

        # Get cultural discourse vocabulary
        cultural_discourse = cultural_data.get("discourse_vocabulary", [])
        if not cultural_discourse:
            # Fallback to concepts and terminology
            cultural_discourse = cultural_data.get("concepts", []) + cultural_data.get("terminology", [])

        if not cultural_discourse:
            return 0.5

        # Convert cultural discourse to word set
        cultural_words = set()
        for item in cultural_discourse:
            if isinstance(item, str):
                cultural_words.update(item.lower().replace('_', ' ').split())

        # Calculate Jaccard similarity as proxy for cosine similarity
        if not text_words or not cultural_words:
            return 0.0

        intersection = text_words.intersection(cultural_words)
        union = text_words.union(cultural_words)

        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        # Convert to range similar to cosine similarity
        return min(1.0, jaccard_similarity * 2.0)

    def _calculate_value_alignment(self, text: str, culture: str, cultural_data: Dict[str, Any]) -> float:
        """
        Calculate Value Alignment (VA) dimension.

        Formula: f_VA(x, A_c) = μ₁·Values(x, V_c) + μ₂·Ethics(x, E_c) + μ₃·Sacred(x, S_c)
        where μ₁=0.4, μ₂=0.35, μ₃=0.25
        """
        if not cultural_data:
            return 0.5

        mu1, mu2, mu3 = 0.4, 0.35, 0.25

        # Values(x, V_c): Alignment with cultural value systems
        value_score = self._compute_value_alignment(text, culture, cultural_data)

        # Ethics(x, E_c): Adherence to cultural ethical frameworks
        ethics_score = self._compute_ethical_alignment(text, culture, cultural_data)

        # Sacred(x, S_c): Respect for sacred boundaries and epistemological restrictions
        sacred_score = self._compute_sacred_boundary_respect(text, culture)

        va_score = mu1 * value_score + mu2 * ethics_score + mu3 * sacred_score
        return min(1.0, max(0.0, va_score))

    def _compute_value_alignment(self, text: str, culture: str, cultural_data: Dict[str, Any]) -> float:
        """Compute alignment with cultural value systems."""
        # Get cultural values from knowledge base
        cultural_values = cultural_data.get("values", [])
        if not cultural_values:
            cultural_values = ["community", "respect", "harmony", "tradition", "wisdom"]

        text_lower = text.lower()
        value_matches = 0

        # Check for explicit value expressions
        for value in cultural_values:
            value_patterns = [
                value.lower(),
                f"respect for {value.lower()}",
                f"importance of {value.lower()}",
                f"value of {value.lower()}"
            ]

            for pattern in value_patterns:
                if pattern in text_lower:
                    value_matches += 1
                    break

        # Check for counter-cultural value expressions
        counter_values = ["individualism", "materialism", "exploitation", "disrespect"]
        counter_matches = 0

        for counter_value in counter_values:
            if counter_value in text_lower:
                counter_matches += 1

        # Calculate alignment score
        if not cultural_values:
            return 0.5

        positive_score = value_matches / len(cultural_values)
        negative_penalty = min(0.5, counter_matches * 0.2)

        return max(0.0, positive_score - negative_penalty)

    def _compute_ethical_alignment(self, text: str, culture: str, cultural_data: Dict[str, Any]) -> float:
        """Compute adherence to cultural ethical frameworks."""
        # Check for culturally inappropriate language
        inappropriate_patterns = [
            r"\bprimitive\b", r"\bbackward\b", r"\bundeveloped\b",
            r"\bsavage\b", r"\buncivilized\b", r"\bexotic\b",
            r"\bmystic\b", r"\bstrange\b", r"\bweird\b"
        ]

        text_lower = text.lower()
        violations = sum(1 for pattern in inappropriate_patterns if re.search(pattern, text_lower))

        # Check for respectful language
        respectful_patterns = [
            r"\btraditional\b", r"\bancestr\w+\b", r"\bwisdom\b",
            r"\bheritage\b", r"\bcultur\w+\b", r"\bhonor\w*\b"
        ]

        respectful_matches = sum(1 for pattern in respectful_patterns if re.search(pattern, text_lower))

        # Calculate ethical alignment
        base_score = 1.0
        violation_penalty = min(0.8, violations * 0.25)
        respectful_bonus = min(0.2, respectful_matches * 0.1)

        return max(0.0, base_score - violation_penalty + respectful_bonus)

    def _compute_sacred_boundary_respect(self, text: str, culture: str) -> float:
        """Compute respect for sacred boundaries and epistemological restrictions."""
        contains_sacred, violations = self.kb.contains_sacred_content(text, culture)

        # Base score assumes respectful handling
        base_score = 1.0

        if contains_sacred:
            # Check if sacred content is handled appropriately
            appropriate_markers = [
                "according to tradition", "as taught by elders", "respectfully",
                "with permission", "as shared publicly", "in accordance with custom"
            ]

            text_lower = text.lower()
            has_appropriate_markers = any(marker in text_lower for marker in appropriate_markers)

            if has_appropriate_markers:
                # Reduced penalty for appropriately contextualized sacred content
                violation_penalty = min(0.3, len(violations) * 0.1)
            else:
                # Higher penalty for inappropriate handling
                violation_penalty = min(0.8, len(violations) * 0.3)

            base_score -= violation_penalty

        return max(0.0, base_score)

    def _calculate_transmission_appropriateness(self,
                                              text: str,
                                              expected_elements: Dict[str, Any],
                                              reference_text: Optional[str] = None) -> float:
        """
        Calculate Transmission Appropriateness (TA) dimension.

        Formula: f_TA(x, T_c) = ν₁·Complete(x, K_c) + ν₂·Pedagogy(x, M_c) + ν₃·Access(x, A_c)
        where ν₁=0.4, ν₂=0.35, ν₃=0.25
        """
        if not expected_elements:
            return 0.5

        nu1, nu2, nu3 = 0.4, 0.35, 0.25

        # Complete(x, K_c): Completeness of knowledge transfer
        completeness_score = self._compute_knowledge_completeness(text, expected_elements)

        # Pedagogy(x, M_c): Appropriateness of pedagogical method
        pedagogy_score = self._compute_pedagogical_appropriateness(text, expected_elements)

        # Access(x, A_c): Accessibility to target learning community
        accessibility_score = self._compute_accessibility(text, expected_elements)

        ta_score = nu1 * completeness_score + nu2 * pedagogy_score + nu3 * accessibility_score
        return min(1.0, max(0.0, ta_score))

    def _compute_knowledge_completeness(self, text: str, expected_elements: Dict[str, Any]) -> float:
        """Compute completeness of knowledge transfer."""
        text_lower = text.lower()
        completeness_score = 0.0
        total_elements = 0

        # Check for expected cultural elements
        for element_type, elements in expected_elements.items():
            if isinstance(elements, list):
                total_elements += len(elements)
                for element in elements:
                    element_str = str(element).lower()
                    # More sophisticated matching including partial matches
                    if self._check_element_presence(element_str, text_lower):
                        completeness_score += 1
            elif isinstance(elements, str):
                total_elements += 1
                element_str = elements.lower()
                if self._check_element_presence(element_str, text_lower):
                    completeness_score += 1

        if total_elements == 0:
            return 0.5

        return completeness_score / total_elements

    def _check_element_presence(self, element: str, text: str) -> bool:
        """Check if cultural element is present in text with various forms."""
        element_variants = [
            element,
            element.replace('_', ' '),
            element.replace('_', ''),
            element.replace(' ', '_'),
            element.capitalize()
        ]

        return any(variant in text for variant in element_variants)

    def _compute_pedagogical_appropriateness(self, text: str, expected_elements: Dict[str, Any]) -> float:
        """Compute appropriateness of pedagogical method for cultural transmission."""
        text_lower = text.lower()

        # Check for traditional transmission markers
        traditional_markers = [
            "story", "narrative", "tale", "legend", "myth",
            "teaching", "lesson", "wisdom", "passed down",
            "generation", "elder", "ancestor", "tradition"
        ]

        marker_count = sum(1 for marker in traditional_markers if marker in text_lower)
        marker_score = min(1.0, marker_count / 5.0)

        # Check for appropriate learning structures
        learning_structures = [
            "example", "demonstration", "practice", "repetition",
            "question", "answer", "dialogue", "interaction"
        ]

        structure_count = sum(1 for structure in learning_structures if structure in text_lower)
        structure_score = min(1.0, structure_count / 4.0)

        # Check for cultural pedagogical patterns
        cultural_patterns = [
            "oral tradition", "hands-on learning", "observation",
            "apprenticeship", "community learning", "ritual practice"
        ]

        pattern_count = sum(1 for pattern in cultural_patterns if pattern in text_lower)
        pattern_score = min(1.0, pattern_count / 3.0)

        # Weighted combination
        return 0.4 * marker_score + 0.35 * structure_score + 0.25 * pattern_score

    def _compute_accessibility(self, text: str, expected_elements: Dict[str, Any]) -> float:
        """Compute accessibility to target learning community."""
        text_lower = text.lower()

        # Check language complexity (simplified measure)
        words = text_lower.split()
        if not words:
            return 0.0

        # Average word length as complexity proxy
        avg_word_length = sum(len(word) for word in words) / len(words)
        complexity_score = max(0.0, 1.0 - (avg_word_length - 5.0) / 10.0)

        # Check for inclusive language
        inclusive_markers = [
            "we", "our", "together", "community", "everyone",
            "all", "shared", "common", "collective"
        ]

        inclusive_count = sum(1 for marker in inclusive_markers if marker in text_lower)
        inclusive_score = min(1.0, inclusive_count / 5.0)

        # Check for age-appropriate content markers
        age_appropriate = [
            "learn", "understand", "discover", "explore",
            "simple", "clear", "step by step", "gradual"
        ]

        age_count = sum(1 for marker in age_appropriate if marker in text_lower)
        age_score = min(1.0, age_count / 4.0)

        # Check for cultural accessibility
        cultural_context = expected_elements.get("cultural_context", {})
        if "community_oriented" in str(cultural_context).lower():
            community_bonus = 0.2 if any(word in text_lower for word in ["community", "together", "shared"]) else 0.0
        else:
            community_bonus = 0.0

        # Weighted combination
        accessibility = 0.3 * complexity_score + 0.3 * inclusive_score + 0.25 * age_score + 0.15 + community_bonus
        return min(1.0, accessibility)

    def _calculate_conceptual_preservation(self, source_text: str, translated_text: str, source_culture: str) -> float:
        """
        Calculate Conceptual Preservation (CP) dimension.

        Formula: f_CP(s, t, C_s) = ω₁·Semantic(s, t) + ω₂·Cultural(s, t, C_s) + ω₃·Concept(s, t, C_s)
        where ω₁=0.4, ω₂=0.35, ω₃=0.25
        """
        if not source_text or not translated_text:
            return 0.0

        omega1, omega2, omega3 = 0.4, 0.35, 0.25

        # Semantic(s, t): Core semantic similarity using embeddings approximation
        semantic_score = self._compute_semantic_similarity(source_text, translated_text)

        # Cultural(s, t, C_s): Culture-bound concept preservation
        cultural_score = self._compute_cultural_concept_preservation(source_text, translated_text, source_culture)

        # Concept(s, t, C_s): Domain-specific concept integrity
        concept_score = self._compute_concept_integrity(source_text, translated_text, source_culture)

        cp_score = omega1 * semantic_score + omega2 * cultural_score + omega3 * concept_score
        return min(1.0, max(0.0, cp_score))

    def _compute_semantic_similarity(self, source_text: str, translated_text: str) -> float:
        """Compute core semantic similarity using embedding approximation."""
        # Use TF-IDF style scoring as BERT embedding approximation
        source_words = source_text.lower().split()
        translated_words = translated_text.lower().split()

        if not source_words or not translated_words:
            return 0.0

        # Create word frequency vectors
        source_freq = {}
        translated_freq = {}

        for word in source_words:
            source_freq[word] = source_freq.get(word, 0) + 1
        for word in translated_words:
            translated_freq[word] = translated_freq.get(word, 0) + 1

        # Normalize frequencies
        source_total = len(source_words)
        translated_total = len(translated_words)

        for word in source_freq:
            source_freq[word] /= source_total
        for word in translated_freq:
            translated_freq[word] /= translated_total

        # Calculate cosine similarity approximation
        all_words = set(source_freq.keys()) | set(translated_freq.keys())
        dot_product = 0.0
        source_norm = 0.0
        translated_norm = 0.0

        for word in all_words:
            s_freq = source_freq.get(word, 0.0)
            t_freq = translated_freq.get(word, 0.0)

            dot_product += s_freq * t_freq
            source_norm += s_freq * s_freq
            translated_norm += t_freq * t_freq

        if source_norm == 0.0 or translated_norm == 0.0:
            return 0.0

        cosine_sim = dot_product / (np.sqrt(source_norm) * np.sqrt(translated_norm))
        return max(0.0, cosine_sim)

    def _compute_cultural_concept_preservation(self, source_text: str, translated_text: str, source_culture: str) -> float:
        """Compute preservation of culture-bound concepts."""
        source_cultural_data = self.kb.get_cultural_concepts(source_culture)
        if not source_cultural_data:
            return 0.5

        source_lower = source_text.lower()
        translated_lower = translated_text.lower()

        # Extract cultural concepts from source
        cultural_concepts = source_cultural_data.get("concepts", [])
        cultural_terms = source_cultural_data.get("terminology", [])
        all_cultural_items = cultural_concepts + cultural_terms

        if not all_cultural_items:
            return 0.5

        preserved_count = 0
        total_found = 0

        for concept in all_cultural_items:
            concept_variants = [
                concept.lower(),
                concept.replace('_', ' ').lower(),
                concept.replace('_', '').lower()
            ]

            # Check if concept appears in source
            found_in_source = any(variant in source_lower for variant in concept_variants)

            if found_in_source:
                total_found += 1
                # Check if preserved in translation (directly or through explanation)
                found_in_translation = any(variant in translated_lower for variant in concept_variants)

                # Check for explanatory preservation
                explanatory_markers = [
                    f"concept of {concept.replace('_', ' ')}",
                    f"traditional {concept.replace('_', ' ')}",
                    f"cultural {concept.replace('_', ' ')}",
                    f"meaning of {concept.replace('_', ' ')}"
                ]

                has_explanation = any(marker in translated_lower for marker in explanatory_markers)

                if found_in_translation or has_explanation:
                    preserved_count += 1

        if total_found == 0:
            return 0.5

        return preserved_count / total_found

    def _compute_concept_integrity(self, source_text: str, translated_text: str, source_culture: str) -> float:
        """Compute domain-specific concept integrity."""
        # Check for domain-specific terminology preservation
        source_cultural_data = self.kb.get_cultural_concepts(source_culture)
        if not source_cultural_data:
            return 0.5

        domains = source_cultural_data.get("domains", {})
        if not domains:
            return 0.5

        total_score = 0.0
        domain_count = 0

        for domain, domain_concepts in domains.items():
            if not domain_concepts:
                continue

            domain_count += 1
            source_lower = source_text.lower()
            translated_lower = translated_text.lower()

            # Count domain concepts in source and check preservation
            domain_terms_in_source = 0
            preserved_terms = 0

            for concept in domain_concepts:
                concept_lower = concept.lower().replace('_', ' ')
                if concept_lower in source_lower:
                    domain_terms_in_source += 1

                    # Check various forms of preservation
                    preservation_patterns = [
                        concept_lower,
                        f"term {concept_lower}",
                        f"concept of {concept_lower}",
                        f"traditional {concept_lower}",
                        concept.lower()  # Original form
                    ]

                    if any(pattern in translated_lower for pattern in preservation_patterns):
                        preserved_terms += 1

            if domain_terms_in_source > 0:
                domain_score = preserved_terms / domain_terms_in_source
                total_score += domain_score

        if domain_count == 0:
            return 0.5

        return total_score / domain_count

    def _calculate_metaphorical_mapping(self,
                                       source_text: str,
                                       translated_text: str,
                                       source_culture: str,
                                       target_culture: str) -> float:
        """
        Calculate Metaphorical Mapping (MM) dimension.

        Formula: f_MM(s, t, C_s, C_t) = ρ₁·Metaphor(s, t) + ρ₂·Figure(s, t) + ρ₃·System(s, t, C_s, C_t)
        where ρ₁=0.4, ρ₂=0.35, ρ₃=0.25
        """
        if not source_text or not translated_text:
            return 0.0

        rho1, rho2, rho3 = 0.4, 0.35, 0.25

        # Metaphor(s, t): Metaphorical transfer accuracy
        metaphor_score = self._compute_metaphor_transfer(source_text, translated_text, source_culture)

        # Figure(s, t): Figurative language preservation
        figurative_score = self._compute_figurative_preservation(source_text, translated_text)

        # System(s, t, C_s, C_t): Conceptual metaphor system mapping
        system_score = self._compute_conceptual_system_mapping(source_text, translated_text, source_culture, target_culture)

        mm_score = rho1 * metaphor_score + rho2 * figurative_score + rho3 * system_score
        return min(1.0, max(0.0, mm_score))

    def _compute_metaphor_transfer(self, source_text: str, translated_text: str, source_culture: str) -> float:
        """Compute metaphorical transfer accuracy."""
        source_cultural_data = self.kb.get_cultural_concepts(source_culture)
        if not source_cultural_data:
            return 0.5

        source_lower = source_text.lower()
        translated_lower = translated_text.lower()

        # Common metaphorical patterns
        metaphorical_patterns = [
            r'\b\w+ is like \w+\b',
            r'\b\w+ as \w+ as \w+\b',
            r'\blike a \w+\b',
            r'\bas if \w+\b',
            r'\bsimilar to \w+\b'
        ]

        # Cultural metaphors from source culture
        cultural_metaphors = source_cultural_data.get("metaphors", [])
        if not cultural_metaphors:
            # Use concepts as potential metaphorical sources
            cultural_metaphors = source_cultural_data.get("concepts", [])

        metaphor_preservation = 0.0
        total_metaphors = 0

        # Check pattern-based metaphors
        for pattern in metaphorical_patterns:
            import re
            source_matches = re.findall(pattern, source_lower)
            translated_matches = re.findall(pattern, translated_lower)

            if source_matches:
                total_metaphors += len(source_matches)
                # Check if similar metaphorical structure is preserved
                if translated_matches:
                    metaphor_preservation += min(len(translated_matches), len(source_matches))

        # Check cultural metaphor preservation
        for metaphor in cultural_metaphors:
            metaphor_variants = [
                metaphor.lower(),
                metaphor.replace('_', ' ').lower()
            ]

            found_in_source = any(variant in source_lower for variant in metaphor_variants)
            if found_in_source:
                total_metaphors += 1

                # Check for direct preservation or equivalent metaphor
                found_in_translation = any(variant in translated_lower for variant in metaphor_variants)

                # Check for metaphorical explanation
                explanation_patterns = [
                    f"like {metaphor.replace('_', ' ').lower()}",
                    f"as {metaphor.replace('_', ' ').lower()}",
                    f"metaphor of {metaphor.replace('_', ' ').lower()}"
                ]

                has_explanation = any(pattern in translated_lower for pattern in explanation_patterns)

                if found_in_translation or has_explanation:
                    metaphor_preservation += 1

        if total_metaphors == 0:
            return 0.5

        return metaphor_preservation / total_metaphors

    def _compute_figurative_preservation(self, source_text: str, translated_text: str) -> float:
        """Compute figurative language preservation."""
        source_lower = source_text.lower()
        translated_lower = translated_text.lower()

        # Figurative language indicators
        figurative_markers = [
            'metaphor', 'simile', 'analogy', 'symbolize', 'represent',
            'embody', 'exemplify', 'personify', 'allegor'
        ]

        # Figurative constructions
        figurative_constructions = [
            r'\blike\b', r'\bas\b', r'\bsuch as\b', r'\bsimilar to\b',
            r'\bcompare to\b', r'\bresemble\b', r'\bmirror\b'
        ]

        source_figurative_count = 0
        translated_figurative_count = 0

        # Count figurative markers
        for marker in figurative_markers:
            source_figurative_count += source_lower.count(marker)
            translated_figurative_count += translated_lower.count(marker)

        # Count figurative constructions
        import re
        for construction in figurative_constructions:
            source_figurative_count += len(re.findall(construction, source_lower))
            translated_figurative_count += len(re.findall(construction, translated_lower))

        if source_figurative_count == 0:
            return 1.0 if translated_figurative_count == 0 else 0.8

        # Calculate preservation ratio
        preservation_ratio = translated_figurative_count / source_figurative_count

        # Penalize excessive loss or gain
        if preservation_ratio > 1.5:
            return 0.7  # Too much figurative language added
        elif preservation_ratio < 0.3:
            return 0.3  # Too much figurative language lost
        else:
            return min(1.0, preservation_ratio)

    def _compute_conceptual_system_mapping(self, source_text: str, translated_text: str, source_culture: str, target_culture: str) -> float:
        """Compute conceptual metaphor system mapping between cultures."""
        source_cultural_data = self.kb.get_cultural_concepts(source_culture)
        target_cultural_data = self.kb.get_cultural_concepts(target_culture) if target_culture else {}

        if not source_cultural_data:
            return 0.5

        source_lower = source_text.lower()
        translated_lower = translated_text.lower()

        # Source conceptual domains
        source_domains = source_cultural_data.get("domains", {})
        target_domains = target_cultural_data.get("domains", {}) if target_cultural_data else {}

        mapping_score = 0.0
        total_mappings = 0

        # Check conceptual domain mappings
        for domain_name, domain_concepts in source_domains.items():
            if not domain_concepts:
                continue

            # Check if source domain concepts appear in source text
            domain_in_source = False
            for concept in domain_concepts:
                concept_lower = concept.lower().replace('_', ' ')
                if concept_lower in source_lower:
                    domain_in_source = True
                    break

            if domain_in_source:
                total_mappings += 1

                # Check if appropriate mapping exists in translation
                # Direct preservation
                domain_preserved = False
                for concept in domain_concepts:
                    concept_lower = concept.lower().replace('_', ' ')
                    if concept_lower in translated_lower:
                        domain_preserved = True
                        break

                # Cross-cultural mapping
                if not domain_preserved and target_domains:
                    # Check if equivalent domain concepts appear
                    for target_domain_name, target_concepts in target_domains.items():
                        if target_domain_name == domain_name or "similar" in target_domain_name:
                            for target_concept in target_concepts:
                                target_concept_lower = target_concept.lower().replace('_', ' ')
                                if target_concept_lower in translated_lower:
                                    domain_preserved = True
                                    break
                            if domain_preserved:
                                break

                # Explanatory mapping
                if not domain_preserved:
                    explanation_patterns = [
                        f"cultural concept of {domain_name}",
                        f"traditional {domain_name}",
                        f"equivalent to {domain_name}",
                        f"similar to {domain_name}"
                    ]
                    domain_preserved = any(pattern in translated_lower for pattern in explanation_patterns)

                if domain_preserved:
                    mapping_score += 1

        if total_mappings == 0:
            return 0.5

        return mapping_score / total_mappings

    def _calculate_pragmatic_accuracy(self,
                                     source_text: str,
                                     translated_text: str,
                                     source_culture: str,
                                     target_culture: str) -> float:
        """
        Calculate Pragmatic Accuracy (PA) dimension.

        Formula: f_PA(s, t, C_s, C_t) = σ₁·Intent(s, t) + σ₂·Social(s, t, C_t) + σ₃·Register(s, t, C_t)
        where σ₁=0.4, σ₂=0.35, σ₃=0.25
        """
        if not source_text or not translated_text:
            return 0.0

        sigma1, sigma2, sigma3 = 0.4, 0.35, 0.25

        # Intent(s, t): Illocutionary force preservation
        intent_score = self._compute_illocutionary_force(source_text, translated_text)

        # Social(s, t, C_t): Socio-pragmatic appropriateness for target culture
        social_score = self._compute_sociopragmatic_appropriateness(source_text, translated_text, target_culture)

        # Register(s, t, C_t): Register and formality level alignment
        register_score = self._compute_register_alignment(source_text, translated_text, target_culture)

        pa_score = sigma1 * intent_score + sigma2 * social_score + sigma3 * register_score
        return min(1.0, max(0.0, pa_score))

    def _compute_illocutionary_force(self, source_text: str, translated_text: str) -> float:
        """Compute preservation of illocutionary force (speech acts)."""
        source_lower = source_text.lower()
        translated_lower = translated_text.lower()

        # Speech act indicators
        speech_acts = {
            'request': [r'\bplease\b', r'\bcould you\b', r'\bwould you\b', r'\bcan you\b'],
            'command': [r'\bmust\b', r'\bshould\b', r'\bneed to\b', r'\bhave to\b'],
            'question': [r'\bwhat\b', r'\bwhy\b', r'\bhow\b', r'\bwhen\b', r'\bwhere\b', r'\?'],
            'assertion': [r'\bis\b', r'\bare\b', r'\bwill be\b', r'\bfact\b'],
            'promise': [r'\bwill\b', r'\bpromise\b', r'\bguarantee\b', r'\bcommit\b'],
            'warning': [r'\bwarning\b', r'\bbeware\b', r'\bcareful\b', r'\bdanger\b']
        }

        source_acts = {}
        translated_acts = {}

        import re
        for act_type, patterns in speech_acts.items():
            source_count = sum(len(re.findall(pattern, source_lower)) for pattern in patterns)
            translated_count = sum(len(re.findall(pattern, translated_lower)) for pattern in patterns)

            source_acts[act_type] = source_count
            translated_acts[act_type] = translated_count

        # Calculate preservation score
        total_source_acts = sum(source_acts.values())
        if total_source_acts == 0:
            return 1.0

        preservation_score = 0.0
        for act_type in speech_acts:
            if source_acts[act_type] > 0:
                # Check if similar number of acts preserved
                preservation_ratio = translated_acts[act_type] / source_acts[act_type]
                # Penalize excessive loss or gain
                if 0.5 <= preservation_ratio <= 1.5:
                    preservation_score += source_acts[act_type]
                elif preservation_ratio > 0:
                    preservation_score += source_acts[act_type] * 0.5

        return preservation_score / total_source_acts

    def _compute_sociopragmatic_appropriateness(self, source_text: str, translated_text: str, target_culture: str) -> float:
        """Compute socio-pragmatic appropriateness for target culture."""
        target_cultural_data = self.kb.get_cultural_concepts(target_culture) if target_culture else {}
        if not target_cultural_data:
            return 0.5

        translated_lower = translated_text.lower()

        # Cultural politeness and social appropriateness
        appropriateness_score = 0.0
        total_checks = 0

        # Check for culturally appropriate expressions
        target_social_norms = target_cultural_data.get("social_norms", [])
        if target_social_norms:
            total_checks += 1
            appropriate_expressions = 0
            for norm in target_social_norms:
                norm_indicators = [
                    norm.lower(),
                    f"respect for {norm.lower()}",
                    f"consideration of {norm.lower()}"
                ]
                if any(indicator in translated_lower for indicator in norm_indicators):
                    appropriate_expressions += 1
            if target_social_norms:
                appropriateness_score += appropriate_expressions / len(target_social_norms)

        # Check for respectful language
        respectful_markers = [
            'respectfully', 'humbly', 'gratefully', 'honored',
            'privilege', 'courtesy', 'kindly', 'graciously'
        ]
        total_checks += 1
        respectful_count = sum(1 for marker in respectful_markers if marker in translated_lower)
        respectful_score = min(1.0, respectful_count / 3.0)
        appropriateness_score += respectful_score

        # Check for inappropriate content
        inappropriate_markers = [
            'rude', 'disrespectful', 'offensive', 'insulting',
            'inappropriate', 'improper', 'unacceptable'
        ]
        inappropriate_count = sum(1 for marker in inappropriate_markers if marker in translated_lower)
        inappropriate_penalty = min(0.5, inappropriate_count * 0.2)
        appropriateness_score -= inappropriate_penalty

        # Check for cultural sensitivity
        total_checks += 1
        sensitivity_markers = [
            'cultural', 'traditional', 'respectful', 'appropriate',
            'sensitive', 'aware', 'understanding'
        ]
        sensitivity_count = sum(1 for marker in sensitivity_markers if marker in translated_lower)
        sensitivity_score = min(1.0, sensitivity_count / 4.0)
        appropriateness_score += sensitivity_score

        if total_checks == 0:
            return 0.5

        return max(0.0, appropriateness_score / total_checks)

    def _compute_register_alignment(self, source_text: str, translated_text: str, target_culture: str) -> float:
        """Compute register and formality level alignment."""
        source_lower = source_text.lower()
        translated_lower = translated_text.lower()

        # Formal language indicators
        formal_indicators = [
            'furthermore', 'moreover', 'consequently', 'therefore',
            'nevertheless', 'however', 'thus', 'hence'
        ]

        # Informal language indicators
        informal_indicators = [
            'yeah', 'ok', 'cool', 'awesome', 'nice',
            'stuff', 'things', 'guy', 'folks'
        ]

        # Academic/technical indicators
        academic_indicators = [
            'analysis', 'research', 'study', 'examination',
            'investigation', 'methodology', 'findings', 'conclusion'
        ]

        # Calculate register scores
        source_formal = sum(1 for indicator in formal_indicators if indicator in source_lower)
        translated_formal = sum(1 for indicator in formal_indicators if indicator in translated_lower)

        source_informal = sum(1 for indicator in informal_indicators if indicator in source_lower)
        translated_informal = sum(1 for indicator in informal_indicators if indicator in translated_lower)

        source_academic = sum(1 for indicator in academic_indicators if indicator in source_lower)
        translated_academic = sum(1 for indicator in academic_indicators if indicator in translated_lower)

        # Determine dominant register in source
        source_total = source_formal + source_informal + source_academic
        if source_total == 0:
            return 0.8  # Neutral text

        # Calculate register preservation
        register_alignment = 0.0

        if source_formal > 0:
            formal_ratio = translated_formal / max(1, source_formal)
            register_alignment += (source_formal / source_total) * min(1.0, formal_ratio)

        if source_informal > 0:
            informal_ratio = translated_informal / max(1, source_informal)
            register_alignment += (source_informal / source_total) * min(1.0, informal_ratio)

        if source_academic > 0:
            academic_ratio = translated_academic / max(1, source_academic)
            register_alignment += (source_academic / source_total) * min(1.0, academic_ratio)

        # Check for register consistency
        translated_total = translated_formal + translated_informal + translated_academic
        if translated_total > 0:
            # Penalize mixed registers if source was consistent
            dominant_source = max(source_formal, source_informal, source_academic)
            if dominant_source / source_total > 0.7:  # Source was consistent
                dominant_translated = max(translated_formal, translated_informal, translated_academic)
                if dominant_translated / translated_total < 0.5:  # Translation is mixed
                    register_alignment *= 0.8

        return register_alignment

    def _calculate_sacred_knowledge_handling(self, text: str, source_culture: str, target_culture: str) -> float:
        """
        Calculate Sacred Knowledge Handling (SKH) dimension.

        Formula: f_SKH(t, C_s, C_t) = τ₁·Sacred(t, C_s, C_t) + τ₂·Ethics(t, C_s, C_t) + τ₃·Access(t, C_s, C_t)
        where τ₁=0.4, τ₂=0.35, τ₃=0.25
        """
        if not text:
            return 1.0

        tau1, tau2, tau3 = 0.4, 0.35, 0.25

        # Sacred(t, C_s, C_t): Sacred boundary respect
        sacred_score = self._compute_sacred_boundary_respect_translation(text, source_culture, target_culture)

        # Ethics(t, C_s, C_t): Ethical treatment of epistemologically-restricted content
        ethics_score = self._compute_ethical_treatment(text, source_culture, target_culture)

        # Access(t, C_s, C_t): Appropriate access level control
        access_score = self._compute_access_control(text, source_culture, target_culture)

        skh_score = tau1 * sacred_score + tau2 * ethics_score + tau3 * access_score
        return min(1.0, max(0.0, skh_score))

    def _compute_sacred_boundary_respect_translation(self, text: str, source_culture: str, target_culture: str) -> float:
        """Compute sacred boundary respect in translation context."""
        # Check both source and target cultures for sacred content
        source_sacred, source_violations = self.kb.contains_sacred_content(text, source_culture)
        target_sacred, target_violations = self.kb.contains_sacred_content(text, target_culture) if target_culture else (False, [])

        contains_sacred = source_sacred or target_sacred
        all_violations = source_violations + target_violations

        if not contains_sacred:
            return 1.0  # Perfect score if no sacred content

        # Base score assumes violations exist
        base_score = 1.0

        # Penalty for each violation
        violation_penalty = min(0.8, len(all_violations) * 0.2)
        base_score -= violation_penalty

        # Check for appropriate sacred content handling
        text_lower = text.lower()
        appropriate_handling = [
            "with permission", "as publicly shared", "according to tradition",
            "respectfully shared", "with cultural guidance", "appropriately contextualized"
        ]

        has_appropriate_context = any(phrase in text_lower for phrase in appropriate_handling)
        if has_appropriate_context:
            base_score += 0.3  # Bonus for appropriate contextualization

        return max(0.0, base_score)

    def _compute_ethical_treatment(self, text: str, source_culture: str, target_culture: str) -> float:
        """Compute ethical treatment of epistemologically-restricted content."""
        text_lower = text.lower()

        # Check for ethical treatment markers
        ethical_markers = [
            "respectfully", "ethically", "appropriately", "responsibly",
            "with care", "sensitively", "mindfully", "thoughtfully"
        ]

        ethical_treatment_count = sum(1 for marker in ethical_markers if marker in text_lower)
        ethical_score = min(1.0, ethical_treatment_count / 3.0)

        # Check for exploitative language
        exploitative_patterns = [
            "exotic", "mysterious", "primitive", "secret ritual",
            "forbidden knowledge", "ancient secret", "hidden wisdom"
        ]

        exploitation_count = sum(1 for pattern in exploitative_patterns if pattern in text_lower)
        exploitation_penalty = min(0.6, exploitation_count * 0.3)

        # Check for proper attribution and consent
        attribution_markers = [
            "with permission", "shared by", "taught by", "according to",
            "as told by", "with guidance from", "with consent"
        ]

        has_attribution = any(marker in text_lower for marker in attribution_markers)
        attribution_bonus = 0.2 if has_attribution else 0.0

        # Check for warning about sensitive content
        warning_markers = [
            "sensitive content", "cultural content", "traditional knowledge",
            "respectful approach", "cultural sensitivity"
        ]

        has_warning = any(marker in text_lower for marker in warning_markers)
        warning_bonus = 0.1 if has_warning else 0.0

        total_score = ethical_score - exploitation_penalty + attribution_bonus + warning_bonus
        return max(0.0, min(1.0, total_score))

    def _compute_access_control(self, text: str, source_culture: str, target_culture: str) -> float:
        """Compute appropriate access level control."""
        text_lower = text.lower()

        # Check for appropriate access level indicators
        public_indicators = [
            "publicly shared", "openly taught", "community knowledge",
            "general tradition", "widely known", "common practice"
        ]

        restricted_indicators = [
            "sacred", "private", "restricted", "ceremonial",
            "initiation", "elder knowledge", "special permission"
        ]

        # Count indicators
        public_count = sum(1 for indicator in public_indicators if indicator in text_lower)
        restricted_count = sum(1 for indicator in restricted_indicators if indicator in text_lower)

        # If restricted content is present, check for appropriate handling
        if restricted_count > 0:
            # Check for appropriate restrictions or warnings
            restriction_handling = [
                "general information only", "publicly available aspects",
                "with appropriate limitations", "respectful overview",
                "public elements only", "non-sacred aspects"
            ]

            appropriate_restrictions = sum(1 for phrase in restriction_handling if phrase in text_lower)

            if appropriate_restrictions > 0:
                return 0.8  # Good handling of restricted content
            else:
                # Check if content acknowledges limitations
                limitation_acknowledgments = [
                    "cannot fully", "limited understanding", "partial knowledge",
                    "respectful boundaries", "appropriate limits"
                ]

                has_limitations = any(phrase in text_lower for phrase in limitation_acknowledgments)
                return 0.6 if has_limitations else 0.3  # Poor handling of restricted content

        # If public content, check for accurate representation
        elif public_count > 0:
            # Check for accurate public knowledge representation
            accuracy_indicators = [
                "accurate", "authentic", "traditional", "verified",
                "confirmed", "established", "recognized"
            ]

            accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in text_lower)
            accuracy_score = min(1.0, accuracy_count / 2.0)

            return 0.5 + (accuracy_score * 0.5)  # Base score + accuracy bonus

        else:
            # Neutral content - check for cultural awareness
            awareness_indicators = [
                "cultural", "traditional", "heritage", "respectful",
                "appropriate", "sensitive", "aware"
            ]

            awareness_count = sum(1 for indicator in awareness_indicators if indicator in text_lower)
            return min(1.0, 0.7 + (awareness_count * 0.1))