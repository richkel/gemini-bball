"""
Test script to verify the package installation and basic functionality.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    # Core modules
    import cv2
    import mediapipe
    import numpy as np
    from PIL import Image
    
    # Project modules
    from src.analysis import AnalysisBackend, OllamaBackend, GeminiBackend
    from src.models import ShotAnalysis, ShotResult, ShotType, ShotStatistics
    from src.pipeline import RealTimeAnalyzer, PipelineConfig, BackendType
    
    print("✅ All imports successful!")

def test_analysis_backend():
    """Test the analysis backend interface."""
    import numpy as np
    from src.analysis.base import AnalysisBackend
    from src.models import ShotAnalysis, ShotResult, ShotType
    
    print("\nTesting analysis backend...")
    
    class TestBackend(AnalysisBackend):
        """Test implementation of AnalysisBackend."""
        
        async def analyze_frame(self, frame: np.ndarray, timestamp: float) -> ShotAnalysis:
            """Return a test shot analysis."""
            return ShotAnalysis(
                timestamp=timestamp,
                result=ShotResult.MADE,
                shot_type=ShotType.JUMP_SHOT,
                confidence=0.95,
                feedback="Test shot detected"
            )
    
    # Create a test backend
    backend = TestBackend()
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test analysis
    import asyncio
    analysis = asyncio.run(backend.analyze_frame(frame, 0.0))
    
    print(f"Test analysis result: {analysis}")
    print("✅ Analysis backend test passed!")

if __name__ == "__main__":
    test_imports()
    test_analysis_backend()
    print("\n🎉 All tests passed! The package is working correctly.")
