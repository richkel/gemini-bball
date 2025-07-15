"""
Data models for basketball shot analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime

__all__ = ['ShotResult', 'ShotType', 'ShotAnalysis', 'ShotStatistics']

class ShotResult(str, Enum):
    """Possible shot results."""
    MADE = "made"
    MISSED = "missed"
    BLOCKED = "blocked"
    FOUL = "foul"

class ShotType(str, Enum):
    """Types of basketball shots."""
    LAYUP = "layup"
    DUNK = "dunk"
    HOOK = "hook"
    JUMP_SHOT = "jump_shot"  # Also accepts "jump shot"
    THREE_POINTER = "three_pointer"
    FREE_THROW = "free_throw"
    FLOATER = "floater"
    TIP_IN = "tip_in"
    
    @classmethod
    def from_string(cls, value: str) -> 'ShotType':
        """Convert a string to a ShotType, handling common variations."""
        # Normalize the input string
        value = value.strip().lower().replace(' ', '_')
        
        # Handle common variations
        if value in ['jump shot', 'jumpshot', 'jump_shot']:
            return cls.JUMP_SHOT
        if value in ['3', '3pt', '3_pt', '3_pointer', '3-pointer']:
            return cls.THREE_POINTER
        if value in ['ft', 'free_throw']:
            return cls.FREE_THROW
            
        # Try to match the value directly to an enum member
        for member in cls:
            if value == member.value or value == member.name.lower():
                return member
                
        # If no match, default to JUMP_SHOT
        return cls.JUMP_SHOT

@dataclass
class ShotAnalysis:
    """Analysis result for a single shot attempt."""
    timestamp: float
    result: ShotResult
    shot_type: ShotType
    confidence: float
    feedback: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ShotStatistics:
    """Running statistics for shot analysis."""
    total_shots: int = 0
    made_shots: int = 0
    missed_shots: int = 0
    field_goal_percentage: float = 0.0
    shot_distribution: Dict[ShotType, int] = field(default_factory=dict)
    last_shot_time: Optional[float] = None
    
    def update(self, shot: ShotAnalysis):
        """Update statistics with a new shot."""
        self.total_shots += 1
        
        if shot.result == ShotResult.MADE:
            self.made_shots += 1
        else:
            self.missed_shots += 1
            
        self.field_goal_percentage = (self.made_shots / self.total_shots) * 100 if self.total_shots > 0 else 0.0
        
        # Update shot distribution
        if shot.shot_type not in self.shot_distribution:
            self.shot_distribution[shot.shot_type] = 0
        self.shot_distribution[shot.shot_type] += 1
        
        self.last_shot_time = shot.timestamp
