"""
Test script to verify all imports are working correctly.
"""
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    print("Testing imports...")
    
    # Test main imports
    from main import BasketballShotAnalyzer
    print("✅ Successfully imported BasketballShotAnalyzer from main")
    
    # Test analysis imports
    from analysis.base import AnalysisBackend
    from analysis.ollama_backend import OllamaBackend
    print("✅ Successfully imported analysis backends")
    
    # Test models
    from models import ShotResult, ShotType, ShotAnalysis, ShotStatistics
    print("✅ Successfully imported models")
    
    # Test utils
    from utils.frame_processor import preprocess_frame, draw_shot_result
    print("✅ Successfully imported utils")
    
    print("\n🎉 All imports are working correctly!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    raise
