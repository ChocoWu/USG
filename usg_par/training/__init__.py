"""Training utilities for USG-Par (target construction, loss assembly)."""

from .targets import build_association_targets, build_relation_targets, matches_to_query2gt
from .loss_assembly import compute_multimodal_losses, compute_single_modality_losses

__all__ = [
    "build_relation_targets",
    "build_association_targets",
    "matches_to_query2gt",
    "compute_single_modality_losses",
    "compute_multimodal_losses",
]