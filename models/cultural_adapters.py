
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Set
import numpy as np
import re
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CulturalDomain(Enum):
    """Cultural domain categories"""
    ORAL_TRADITIONS = "oral_traditions"
    PERFORMING_ARTS = "performing_arts"
    SOCIAL_PRACTICES = "social_practices"
    TRADITIONAL_CRAFTS = "traditional_crafts"
    NATURE_KNOWLEDGE = "nature_knowledge"
    SACRED_SPIRITUAL = "sacred_spiritual"


class CulturalAdapter(nn.Module):
  

    def __init__(
        self,
        hidden_size: int,
        num_cultures: int = 147,
        cultural_embedding_dim: int = 512,
        num_domains: int = 6,
        adaptation_strength: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_cultures = num_cultures
        self.cultural_embedding_dim = cultural_embedding_dim
        self.num_domains = num_domains
        self.adaptation_strength = adaptation_strength

        # Cultural embeddings
        self.cultural_embeddings = nn.Embedding(
            num_cultures, cultural_embedding_dim
        )

        # Domain embeddings
        self.domain_embeddings = nn.Embedding(
            num_domains, cultural_embedding_dim
        )

        # Cultural routing network
        self.cultural_router = nn.Sequential(
            nn.Linear(cultural_embedding_dim * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1)
        )

        # Adaptation gates
        self.adaptation_gate = nn.Sequential(
            nn.Linear(hidden_size + cultural_embedding_dim, hidden_size),
            nn.Sigmoid()
        )

        # Cultural bias parameters
        self.cultural_bias = nn.Parameter(
            torch.zeros(num_cultures, hidden_size)
        )

        # Culture mapping
        self.culture_to_id = self._build_culture_mapping()
        self.domain_to_id = self._build_domain_mapping()

        # Initialize weights
        self._initialize_weights()

    def _build_culture_mapping(self) -> Dict[str, int]:
        """Build mapping from culture names to IDs"""
        # This would be loaded from a comprehensive cultural taxonomy
        cultures = [
            # East Asian cultures
            "chinese_han", "japanese", "korean", "mongolian", "tibetan",
            "vietnamese", "thai", "khmer", "lao", "burmese",

            # South Asian cultures
            "hindi", "bengali", "tamil", "telugu", "marathi", "gujarati",
            "punjabi", "kannada", "malayalam", "oriya", "assamese",
            "nepali", "sinhala", "urdu",

            # Southeast Asian cultures
            "javanese", "sundanese", "balinese", "malay", "filipino",
            "bugis", "minangkabau", "batak", "dayak", "iban",

            # Middle Eastern cultures
            "arabic", "persian", "turkish", "kurdish", "armenian",
            "georgian", "azerbaijani", "hebrew", "assyrian",

            # European cultures
            "english", "french", "german", "italian", "spanish",
            "portuguese", "dutch", "scandinavian", "slavic", "celtic",
            "greek", "albanian", "romanian", "hungarian", "basque",

            # African cultures
            "yoruba", "igbo", "hausa", "swahili", "amharic",
            "akan", "ewe", "mandinka", "wolof", "fulani",
            "shona", "zulu", "xhosa", "sotho", "berber",

            # Native American cultures
            "cherokee", "navajo", "lakota", "apache", "pueblo",
            "iroquois", "algonquian", "inuit", "ojibwe", "cree",

            # Latin American cultures
            "mexican", "guatemalan", "peruvian", "bolivian", "ecuadorian",
            "brazilian", "argentinian", "chilean", "colombian", "venezuelan",

            # Pacific Islander cultures
            "hawaiian", "maori", "fijian", "samoan", "tongan",
            "tahitian", "aboriginal_australian", "torres_strait",

            # Central Asian cultures
            "kazakh", "kyrgyz", "uzbek", "tajik", "turkmen",
            "afghan", "baloch", "pashtun",

            # Additional cultures to reach 147
            "romani", "sami", "maltese", "cornish", "manx",
            "faroese", "icelandic", "greenlandic", "aleut", "ainu"
        ]

        # Extend to 147 cultures with regional variants
        extended_cultures = cultures[:]
        for i in range(147 - len(cultures)):
            extended_cultures.append(f"regional_variant_{i}")

        return {culture: idx for idx, culture in enumerate(extended_cultures[:147])}

    def _build_domain_mapping(self) -> Dict[str, int]:
        """Build mapping from cultural domains to IDs"""
        return {
            "oral_traditions": 0,
            "performing_arts": 1,
            "social_practices": 2,
            "traditional_crafts": 3,
            "nature_knowledge": 4,
            "sacred_spiritual": 5
        }

    def _initialize_weights(self):
        """Initialize adaptation weights"""
        # Initialize cultural embeddings with small random values
        nn.init.normal_(self.cultural_embeddings.weight, mean=0, std=0.1)
        nn.init.normal_(self.domain_embeddings.weight, mean=0, std=0.1)

        # Initialize cultural bias to zero
        nn.init.zeros_(self.cultural_bias)

        # Initialize router weights
        for module in self.cultural_router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cultural_context: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply cultural adaptation to hidden states

        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_size]
            cultural_context: Cultural context information

        Returns:
            Adapted hidden states
        """

        batch_size, seq_len, hidden_size = hidden_states.shape

        # Extract cultural information
        culture = cultural_context.get("culture", "generic")
        domain = cultural_context.get("cultural_domain", "oral_traditions")

        # Get cultural and domain IDs
        culture_id = self.culture_to_id.get(culture, 0)  # Default to first culture
        domain_id = self.domain_to_id.get(domain, 0)

        # Get embeddings
        culture_embed = self.cultural_embeddings(
            torch.tensor(culture_id, device=hidden_states.device)
        )
        domain_embed = self.domain_embeddings(
            torch.tensor(domain_id, device=hidden_states.device)
        )

        # Combine cultural and domain embeddings
        combined_embed = torch.cat([culture_embed, domain_embed], dim=0)

        # Generate cultural routing vector
        cultural_routing = self.cultural_router(combined_embed)
        cultural_routing = cultural_routing.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]

        # Broadcast to match hidden states shape
        cultural_routing = cultural_routing.expand(batch_size, seq_len, -1)

        # Compute adaptation gate
        gate_input = torch.cat([hidden_states, cultural_routing], dim=-1)
        adaptation_gate = self.adaptation_gate(gate_input)

        # Apply cultural bias
        cultural_bias = self.cultural_bias[culture_id]
        cultural_bias = cultural_bias.unsqueeze(0).unsqueeze(0)
        cultural_bias = cultural_bias.expand(batch_size, seq_len, -1)

        # Apply adaptation
        cultural_adjustment = cultural_routing + cultural_bias
        adapted_states = hidden_states + self.adaptation_strength * (
            adaptation_gate * cultural_adjustment
        )

        return adapted_states

    def get_cultural_embedding(self, culture: str) -> torch.Tensor:
        """Get cultural embedding for a specific culture"""
        culture_id = self.culture_to_id.get(culture, 0)
        return self.cultural_embeddings(torch.tensor(culture_id))

    def get_cultural_similarity(self, culture1: str, culture2: str) -> float:
        """Compute similarity between two cultures"""
        embed1 = self.get_cultural_embedding(culture1)
        embed2 = self.get_cultural_embedding(culture2)
        similarity = F.cosine_similarity(embed1, embed2, dim=0)
        return similarity.item()

    def list_supported_cultures(self) -> List[str]:
        """Get list of all supported cultures"""
        return list(self.culture_to_id.keys())


class SacredBoundaryFilter(nn.Module):
  

    def __init__(
        self,
        hidden_size: int,
        filter_strength: float = 0.9,
        num_sensitivity_levels: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.filter_strength = filter_strength
        self.num_sensitivity_levels = num_sensitivity_levels

        # Sacred content detection network
        self.sacred_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_sensitivity_levels)
        )

        # Content protection filter
        self.protection_filter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Sensitivity thresholds
        self.sensitivity_thresholds = {
            "public": 0.0,      # No filtering needed
            "restricted": 0.3,   # Light filtering
            "sacred": 0.7,      # Strong filtering
            "forbidden": 0.9    # Maximum filtering/blocking
        }

        # Sacred content indicators by culture
        self.sacred_indicators = self._build_sacred_indicators()

    def _build_sacred_indicators(self) -> Dict[str, Set[str]]:
        """Build sacred content indicators by culture"""
        return {
            "native_american": {
                "sacred_sites", "vision_quest", "sweat_lodge", "medicine_wheel",
                "sacred_pipe", "sundance", "smudging", "spirit_animal",
                "medicine_bundle", "prayer_stick", "sacred_directions"
            },
            "tibetan": {
                "secret_mantra", "tantric", "initiation", "empowerment",
                "dakini", "terma", "guru_yoga", "phowa", "bardo",
                "sacred_dance", "monastery_secrets"
            },
            "aboriginal_australian": {
                "songlines", "dreaming", "sacred_sites", "men_business",
                "women_business", "initiation", "ancestor_spirits",
                "traditional_law", "sacred_objects"
            },
            "hindu": {
                "sacred_thread", "guru_mantra", "initiation_rites",
                "temple_sanctum", "sacred_texts", "tantric_practices",
                "spiritual_lineage", "ashram_rules"
            },
            "islamic": {
                "sacred_names", "quranic_recitation", "sufi_practices",
                "pilgrimage_rituals", "sacred_times", "mosque_protocols"
            },
            "generic": {
                "sacred", "secret", "forbidden", "restricted", "holy",
                "initiation", "spiritual_secret", "ancestral_knowledge",
                "ritual_secret", "clan_knowledge", "family_secret"
            }
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        cultural_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
      

        if cultural_context is None:
            return hidden_states

        # Detect sacred content sensitivity
        sensitivity_logits = self.sacred_detector(hidden_states)
        sensitivity_scores = F.softmax(sensitivity_logits, dim=-1)

        # Determine maximum sensitivity level
        max_sensitivity = torch.max(sensitivity_scores, dim=-1)[0]
        max_sensitivity_level = torch.argmax(sensitivity_scores, dim=-1)

        # Get culture-specific sacred indicators
        culture = cultural_context.get("culture", "generic")
        sacred_indicators = self.sacred_indicators.get(culture, set())

        # Apply protection based on sensitivity level
        protection_mask = self._compute_protection_mask(
            sensitivity_scores,
            max_sensitivity,
            cultural_context
        )

        # Apply protection filter
        if torch.any(protection_mask > 0.5):
            protected_features = self.protection_filter(hidden_states)
            # Blend original and protected features based on mask
            protection_mask = protection_mask.unsqueeze(-1)
            hidden_states = (
                (1 - protection_mask) * hidden_states +
                protection_mask * protected_features
            )

        return hidden_states

    def _compute_protection_mask(
        self,
        sensitivity_scores: torch.Tensor,
        max_sensitivity: torch.Tensor,
        cultural_context: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute protection mask based on sensitivity scores"""

        batch_size, seq_len, _ = sensitivity_scores.shape

        # Base protection mask from sensitivity scores
        forbidden_threshold = self.sensitivity_thresholds["forbidden"]
        sacred_threshold = self.sensitivity_thresholds["sacred"]

        protection_mask = torch.zeros_like(max_sensitivity)

        # Apply strong protection for forbidden content
        forbidden_mask = max_sensitivity > forbidden_threshold
        protection_mask[forbidden_mask] = 1.0

        # Apply medium protection for sacred content
        sacred_mask = (max_sensitivity > sacred_threshold) & ~forbidden_mask
        protection_mask[sacred_mask] = 0.7

        # Adjust based on cultural context
        sacred_boundary_respect = cultural_context.get("sacred_boundary_respect", True)
        if not sacred_boundary_respect:
            protection_mask *= 0.5  # Reduce protection if not explicitly requested

        return protection_mask

    def filter_token_logits(
        self,
        logits: torch.Tensor,
        cultural_context: Dict[str, Any]
    ) -> torch.Tensor:
     

        # Get cultural sacred indicators
        culture = cultural_context.get("culture", "generic")
        sacred_respect = cultural_context.get("sacred_boundary_respect", True)

        if not sacred_respect:
            return logits

        # Apply vocabulary-level filtering
        # This would require a vocabulary mapping to sacred terms
        # For now, apply a general dampening to high-probability tokens
        # that might contain sacred content

        filtered_logits = logits.clone()

        # Reduce probability of tokens that might be problematic
        # In practice, this would use a sacred vocabulary mapping
        top_k = torch.topk(logits, k=100, dim=-1)
        for i, (values, indices) in enumerate(zip(top_k.values, top_k.indices)):
            # Apply conservative filtering to top tokens
            filtered_logits[i, indices] *= 0.9

        return filtered_logits

    def post_process_text(
        self,
        generated_text: str,
        cultural_context: Dict[str, Any]
    ) -> str:
      

        if not cultural_context.get("sacred_boundary_respect", True):
            return generated_text

        culture = cultural_context.get("culture", "generic")
        sacred_indicators = self.sacred_indicators.get(culture, set())

        protected_text = generated_text

        # Replace or warn about sacred terms
        for sacred_term in sacred_indicators:
            pattern = rf'\b{re.escape(sacred_term.replace("_", " "))}\b'
            if re.search(pattern, protected_text, re.IGNORECASE):
                # Replace with respectful placeholder
                replacement = "[SACRED CONTENT - REQUIRES CULTURAL PERMISSION]"
                protected_text = re.sub(pattern, replacement, protected_text, flags=re.IGNORECASE)

        # Add cultural respect disclaimer if sacred content was detected
        if "[SACRED CONTENT" in protected_text:
            disclaimer = (
                "\n\n[Note: Some content has been protected out of respect for "
                f"{culture} cultural sensitivities. Please consult with cultural "
                "experts or community representatives for complete information.]"
            )
            protected_text += disclaimer

        return protected_text

    def assess_content_sensitivity(
        self,
        text: str,
        culture: str
    ) -> Dict[str, Any]:
     

        sacred_indicators = self.sacred_indicators.get(culture, set())
        text_lower = text.lower()

        # Check for sacred indicators
        found_indicators = []
        for indicator in sacred_indicators:
            if indicator.replace("_", " ") in text_lower:
                found_indicators.append(indicator)

        # Determine sensitivity level
        if len(found_indicators) >= 3:
            sensitivity_level = "forbidden"
        elif len(found_indicators) >= 2:
            sensitivity_level = "sacred"
        elif len(found_indicators) >= 1:
            sensitivity_level = "restricted"
        else:
            sensitivity_level = "public"

        return {
            "sensitivity_level": sensitivity_level,
            "sacred_indicators_found": found_indicators,
            "requires_protection": sensitivity_level in ["sacred", "forbidden"],
            "cultural_consultation_recommended": len(found_indicators) > 0,
            "confidence": min(len(found_indicators) / 3, 1.0)
        }


class CulturalKnowledgeRouter(nn.Module):
 

    def __init__(
        self,
        hidden_size: int,
        num_cultures: int = 147,
        num_knowledge_bases: int = 10
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_cultures = num_cultures
        self.num_knowledge_bases = num_knowledge_bases

        # Query classification network
        self.query_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_knowledge_bases),
            nn.Softmax(dim=-1)
        )

        # Cultural routing network
        self.cultural_router = nn.Sequential(
            nn.Linear(hidden_size + num_cultures, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_knowledge_bases),
            nn.Softmax(dim=-1)
        )

        # Knowledge base types
        self.knowledge_bases = [
            "ritual_practices", "traditional_crafts", "oral_traditions",
            "performing_arts", "ecological_knowledge", "social_customs",
            "spiritual_beliefs", "historical_narratives", "linguistic_heritage",
            "material_culture"
        ]

    def forward(
        self,
        query_representation: torch.Tensor,
        cultural_context: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
     
        batch_size = query_representation.shape[0]

        # Create cultural one-hot encoding
        culture = cultural_context.get("culture", "generic")
        # This would use the culture mapping from CulturalAdapter
        culture_id = 0  # Simplified for demo
        cultural_encoding = torch.zeros(batch_size, self.num_cultures)
        cultural_encoding[:, culture_id] = 1.0

        # Classify query type
        query_routing = self.query_classifier(query_representation)

        # Apply cultural routing
        cultural_input = torch.cat([query_representation, cultural_encoding], dim=-1)
        cultural_routing = self.cultural_router(cultural_input)

        # Combine routing signals
        combined_routing = 0.7 * cultural_routing + 0.3 * query_routing

        # Get top knowledge bases
        top_k = torch.topk(combined_routing, k=3, dim=-1)

        return {
            "routing_probabilities": combined_routing,
            "top_knowledge_bases": top_k.indices,
            "top_scores": top_k.values,
            "recommended_sources": [
                self.knowledge_bases[idx.item()]
                for idx in top_k.indices[0]  # First batch item
            ]
        }