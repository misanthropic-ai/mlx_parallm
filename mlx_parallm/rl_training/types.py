from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class Message(TypedDict, total=False):
    role: str
    content: str
    reward: Optional[float]


class ScoredDataGroup(TypedDict, total=False):
    """Represents a group of alternative trajectories for GRPO.
    
    Each group contains multiple trajectories (alternative completions) that share
    the same prompt. The scores represent the final reward for each trajectory.
    """
    tokens: List[List[int]]  # Required: List of token sequences (trajectories)
    masks: List[List[int]]   # Required: List of masks (1 = train, 0 = ignore)
    scores: List[float]       # Required: Final reward for each trajectory
    advantages: Optional[List[List[float]]]     # Per-token advantages if precomputed
    ref_logprobs: Optional[List[List[float]]]   # Per-token ref model logprobs if precomputed
    messages: Optional[List[List[Message]]]     # Message history for each trajectory
    overrides: Optional[List[Dict]]             # Per-trajectory overrides
    group_overrides: Optional[Dict]             # Group-level overrides
    images: Optional[Any]                       # Multimodal data
    env_id: Optional[int]                       # Environment that generated this group

