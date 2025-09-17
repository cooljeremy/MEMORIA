
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CulturalLoss(nn.Module):


    def __init__(self,
                 cultural_weight: float = 0.1,
                 consistency_weight: float = 0.4,
                 boundary_weight: float = 0.3,
                 alignment_weight: float = 0.2,
                 specificity_weight: float = 0.1,
                 num_cultures: int = 147):
        super().__init__()

        self.cultural_weight = cultural_weight
        self.consistency_weight = consistency_weight
        self.boundary_weight = boundary_weight
        self.alignment_weight = alignment_weight
        self.specificity_weight = specificity_weight
        self.num_cultures = num_cultures

        # Initialize cultural consistency loss
        self.consistency_loss = CulturalConsistencyLoss(num_cultures)

        # Initialize sacred boundary loss
        self.boundary_loss = SacredBoundaryLoss()

        # Initialize cross-cultural alignment loss
        self.alignment_loss = CrossCulturalAlignmentLoss(num_cultures)

        # Initialize cultural specificity loss
        self.specificity_loss = CulturalSpecificityLoss()

    def forward(self,
                outputs,
                batch: Dict[str, Any],
                model) -> torch.Tensor:
        """
        Calculate combined cultural loss.

        Args:
            outputs: Model outputs (transformer outputs)
            batch: Training batch with cultural contexts
            model: The model being trained

        Returns:
            Combined cultural loss tensor
        """
        cultural_contexts = batch.get("cultural_contexts", [])
        if not cultural_contexts:
            return torch.tensor(0.0, device=outputs.last_hidden_state.device)

        # Calculate component losses
        consistency_loss = self.consistency_loss(outputs, cultural_contexts, model)
        boundary_loss = self.boundary_loss(outputs, cultural_contexts, model)
        alignment_loss = self.alignment_loss(outputs, cultural_contexts, model)
        specificity_loss = self.specificity_loss(outputs, cultural_contexts, model)

        # Combine losses
        total_cultural_loss = (
            self.consistency_weight * consistency_loss +
            self.boundary_weight * boundary_loss +
            self.alignment_weight * alignment_loss +
            self.specificity_weight * specificity_loss
        )

        return total_cultural_loss


class CulturalConsistencyLoss(nn.Module):
    """
    Loss that encourages cultural consistency within generated text.
    """

    def __init__(self, num_cultures: int = 147, temperature: float = 0.1):
        super().__init__()
        self.num_cultures = num_cultures
        self.temperature = temperature

        # Create culture embeddings for consistency checking
        self.culture_embeddings = nn.Embedding(num_cultures, 512)
        self.culture_to_idx = {}  # Will be populated during training

    def forward(self,
                outputs,
                cultural_contexts: List[Dict[str, Any]],
                model) -> torch.Tensor:
        """Calculate cultural consistency loss."""
        if not cultural_contexts:
            return torch.tensor(0.0, device=outputs.last_hidden_state.device)

        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = hidden_states.shape

        consistency_losses = []

        for i, context in enumerate(cultural_contexts):
            culture = context.get("culture", "unknown")

            if culture == "unknown":
                continue

            # Get culture index
            if culture not in self.culture_to_idx:
                self.culture_to_idx[culture] = len(self.culture_to_idx) % self.num_cultures

            culture_idx = self.culture_to_idx[culture]

            # Get cultural embedding
            culture_emb = self.culture_embeddings(
                torch.tensor(culture_idx, device=hidden_states.device)
            )  # [512]

            # Calculate consistency between hidden states and culture embedding
            sample_hidden = hidden_states[i]  # [seq_len, hidden_size]

            # Project hidden states to culture embedding dimension
            if hidden_size != culture_emb.size(0):
                # Simple linear projection
                proj_weight = torch.randn(
                    hidden_size, culture_emb.size(0),
                    device=hidden_states.device
                ) * 0.02
                sample_hidden_proj = torch.matmul(sample_hidden, proj_weight)
            else:
                sample_hidden_proj = sample_hidden

            # Calculate cosine similarity
            similarities = F.cosine_similarity(
                sample_hidden_proj,  # [seq_len, emb_dim]
                culture_emb.unsqueeze(0).expand_as(sample_hidden_proj),  # [seq_len, emb_dim]
                dim=-1
            )  # [seq_len]

            # Consistency loss: encourage high similarity
            consistency_loss = 1.0 - similarities.mean()
            consistency_losses.append(consistency_loss)

        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, device=outputs.last_hidden_state.device)


class SacredBoundaryLoss(nn.Module):
    """
    Loss that penalizes violations of cultural sacred boundaries.
    """

    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight

        # Sacred boundary indicators (simplified)
        self.sacred_terms = [
            "sacred", "holy", "religious", "spiritual", "ceremonial",
            "ritual", "ancestor", "deity", "temple", "shrine"
        ]

    def forward(self,
                outputs,
                cultural_contexts: List[Dict[str, Any]],
                model) -> torch.Tensor:
        """Calculate sacred boundary violation loss."""
        # This is a simplified implementation
        # In practice, this would use sophisticated cultural knowledge bases

        device = outputs.last_hidden_state.device
        boundary_losses = []

        for i, context in enumerate(cultural_contexts):
            sacred_level = context.get("sacred_level", 0)

            # If content is marked as sacred (level > 0), apply penalty
            if sacred_level > 0:
                # Simple penalty based on sacred level
                penalty = torch.tensor(
                    float(sacred_level) * self.penalty_weight,
                    device=device
                )
                boundary_losses.append(penalty)

        if boundary_losses:
            return torch.stack(boundary_losses).mean()
        else:
            return torch.tensor(0.0, device=device)


class CrossCulturalAlignmentLoss(nn.Module):
    """
    Loss that aligns similar cultural concepts across different cultures.
    """

    def __init__(self, num_cultures: int = 147):
        super().__init__()
        self.num_cultures = num_cultures

        # Create alignment matrix for cultural concepts
        self.alignment_matrix = nn.Parameter(
            torch.randn(num_cultures, num_cultures) * 0.02
        )

    def forward(self,
                outputs,
                cultural_contexts: List[Dict[str, Any]],
                model) -> torch.Tensor:
        """Calculate cross-cultural alignment loss."""
        # Simplified implementation focusing on maintaining
        # semantic similarity across cultures for similar concepts

        hidden_states = outputs.last_hidden_state
        device = hidden_states.device

        # Group samples by cultural domain
        domain_groups = {}
        for i, context in enumerate(cultural_contexts):
            domain = context.get("domain", "other")
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(i)

        alignment_losses = []

        # For samples in the same domain but different cultures,
        # encourage similar representations
        for domain, indices in domain_groups.items():
            if len(indices) < 2:
                continue

            domain_hidden_states = hidden_states[indices]  # [num_samples, seq_len, hidden_size]

            # Calculate pairwise similarities
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    context_i = cultural_contexts[indices[i]]
                    context_j = cultural_contexts[indices[j]]

                    culture_i = context_i.get("culture", "unknown")
                    culture_j = context_j.get("culture", "unknown")

                    if culture_i != culture_j and culture_i != "unknown" and culture_j != "unknown":
                        # Calculate representation similarity
                        repr_i = domain_hidden_states[i].mean(dim=0)  # [hidden_size]
                        repr_j = domain_hidden_states[j].mean(dim=0)  # [hidden_size]

                        similarity = F.cosine_similarity(
                            repr_i.unsqueeze(0),
                            repr_j.unsqueeze(0),
                            dim=1
                        )

                        # Encourage moderate similarity (not identical, but related)
                        target_similarity = torch.tensor(0.7, device=device)
                        alignment_loss = F.mse_loss(similarity, target_similarity)
                        alignment_losses.append(alignment_loss)

        if alignment_losses:
            return torch.stack(alignment_losses).mean()
        else:
            return torch.tensor(0.0, device=device)


class CulturalSpecificityLoss(nn.Module):
    """
    Loss that encourages culture-specific representations over generic ones.
    """

    def __init__(self, specificity_threshold: float = 0.5):
        super().__init__()
        self.specificity_threshold = specificity_threshold

    def forward(self,
                outputs,
                cultural_contexts: List[Dict[str, Any]],
                model) -> torch.Tensor:
        """Calculate cultural specificity loss."""
        device = outputs.last_hidden_state.device
        hidden_states = outputs.last_hidden_state

        specificity_losses = []

        for i, context in enumerate(cultural_contexts):
            culture = context.get("culture", "unknown")
            tradition_type = context.get("tradition_type", "generic")

            if culture == "unknown" or tradition_type == "generic":
                continue

            # Encourage unique representations for specific cultural contexts
            sample_hidden = hidden_states[i]  # [seq_len, hidden_size]

            # Calculate entropy of hidden states as a proxy for specificity
            # Higher entropy suggests more specific, diverse representations
            sample_probs = F.softmax(sample_hidden, dim=-1)
            entropy = -torch.sum(sample_probs * torch.log(sample_probs + 1e-10), dim=-1)
            avg_entropy = entropy.mean()

            # Encourage entropy above threshold
            if avg_entropy < self.specificity_threshold:
                specificity_loss = self.specificity_threshold - avg_entropy
                specificity_losses.append(specificity_loss)

        if specificity_losses:
            return torch.stack(specificity_losses).mean()
        else:
            return torch.tensor(0.0, device=device)