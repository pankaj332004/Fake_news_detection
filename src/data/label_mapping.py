"""
Label taxonomy and mapping definitions for the LIAR benchmark dataset.
Supports 6-way multi-class, 3-way stance, and 2-way binary classifications.
"""

from typing import Dict, List, Optional

# Official LIAR 6-way labels
SIX_WAY_LABELS: List[str] = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

# Integer mappings for 6-class modeling (ordered from most false to most true)
LABEL_TO_ID_6WAY: Dict[str, int] = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

ID_TO_LABEL_6WAY: Dict[int, str] = {v: k for k, v in LABEL_TO_ID_6WAY.items()}

# Binary mapping (0 = Real / Reliable, 1 = Fake / Unreliable)
# In standard literature: pants-fire, false, barely-true are categorized as Fake (1)
# half-true, mostly-true, true are categorized as Real/True-leaning (0)
# An optional 'strict' mode excludes or separates half-true.
BINARY_MAP_DEFAULT: Dict[str, int] = {
    "pants-fire": 1,
    "false": 1,
    "barely-true": 1,
    "half-true": 0,
    "mostly-true": 0,
    "true": 0,
}

BINARY_LABELS: Dict[int, str] = {
    0: "Real",
    1: "Fake",
}

# Rich UI / Visualization metadata
LABEL_METADATA = {
    "pants-fire": {
        "name": "Pants on Fire",
        "description": "Blatantly false, fabricated, or ridiculous claim.",
        "color": "#dc2626",  # Red
        "severity": 5,
        "is_fake": True,
    },
    "false": {
        "name": "False",
        "description": "Factually inaccurate claim.",
        "color": "#ea580c",  # Orange-red
        "severity": 4,
        "is_fake": True,
    },
    "barely-true": {
        "name": "Barely True",
        "description": "Contains an element of truth but ignores critical facts that would give a completely different impression.",
        "color": "#f59e0b",  # Amber
        "severity": 3,
        "is_fake": True,
    },
    "half-true": {
        "name": "Half True",
        "description": "Partially accurate but leaves out important details or takes things out of context.",
        "color": "#eab308",  # Yellow
        "severity": 2,
        "is_fake": False,
    },
    "mostly-true": {
        "name": "Mostly True",
        "description": "Accurate statement, but needs clarification or additional information.",
        "color": "#84cc16",  # Lime
        "severity": 1,
        "is_fake": False,
    },
    "true": {
        "name": "True",
        "description": "Accurate claim with no significant omissions or distortions.",
        "color": "#16a34a",  # Green
        "severity": 0,
        "is_fake": False,
    },
}


def map_to_binary(label: str, exclude_half_true: bool = False) -> Optional[int]:
    """Maps a 6-way label to a binary integer (1=Fake, 0=Real)."""
    norm = str(label).strip().lower()
    if exclude_half_true and norm == "half-true":
        return None
    return BINARY_MAP_DEFAULT.get(norm)


def map_to_6way_id(label: str) -> Optional[int]:
    """Maps a 6-way label to its canonical ID (0-5)."""
    norm = str(label).strip().lower()
    return LABEL_TO_ID_6WAY.get(norm)
