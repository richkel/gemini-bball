"""
Base classes for analysis backends.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np
from ..models.shot_models import ShotAnalysis, ShotStatistics, ShotType, ShotResult

class AnalysisBackend(ABC):
    """Abstract base class for analysis backends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the analysis backend with optional configuration.
        
        Args:
            config: Configuration dictionary for the backend
        """
        self.config = config or {}
        self.stats = ShotStatistics()
        self.shot_history: List[ShotAnalysis] = []
    
    @abstractmethod
    async def analyze_frame(self, frame: np.ndarray, timestamp: float) -> Optional[ShotAnalysis]:
        """Analyze a single frame for shot detection and analysis.
        
        Args:
            frame: Input frame as a numpy array (BGR format)
            timestamp: Timestamp of the frame in seconds
            
        Returns:
            ShotAnalysis object if a shot is detected, None otherwise
        """
        pass
    
    def update_statistics(self, shot: ShotAnalysis) -> ShotStatistics:
        """Update running statistics with a new shot analysis.
        
        Args:
            shot: The shot analysis to add to statistics
            
        Returns:
            Updated ShotStatistics
        """
        self.shot_history.append(shot)
        self.stats.update(shot)
        return self.stats
    
    def get_recent_shots(self, count: int = 5) -> List[ShotAnalysis]:
        """Get the most recent shot analyses.
        
        Args:
            count: Number of recent shots to return
            
        Returns:
            List of ShotAnalysis objects, most recent first
        """
        return self.shot_history[-count:]
    
    def reset_statistics(self) -> None:
        """Reset all statistics and history."""
        self.stats = ShotStatistics()
        self.shot_history = []
    
    @property
    def is_ready(self) -> bool:
        """Check if the backend is ready to process frames."""
        return True
    
    async def close(self) -> None:
        """Clean up any resources used by the backend."""
        pass
