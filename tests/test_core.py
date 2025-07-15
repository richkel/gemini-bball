"""
Test core functionality of the Basketball Shot Analyzer.
"""
import sys
from pathlib import Path
import asyncio
import cv2
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.ollama_backend import OllamaBackend

def create_test_frame(width=640, height=480):
    """Create a test frame with a simple pattern."""
    # Create a simple gradient frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            frame[y, x] = [
                int(255 * x / width),  # Red gradient
                int(255 * y / height),  # Green gradient
                128  # Constant blue
            ]
    return frame

async def test_ollama_backend():
    """Test the Ollama backend with a test frame."""
    print("Initializing Ollama backend...")
    backend = OllamaBackend(model_name="gemma3:12b-it-q4_K_M")
    
    print("Creating test frame...")
    test_frame = create_test_frame()
    
    print("Running analysis...")
    try:
        result = await backend.analyze_frame(test_frame, timestamp=0.0)
        if result:
            print("\nAnalysis Results:")
            print(f"- Shot detected: {result.shot_type}")
            print(f"- Result: {result.result}")
            print(f"- Confidence: {result.confidence:.2f}")
            print(f"- Feedback: {result.feedback}")
        else:
            print("No shot detected in the test frame.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
    finally:
        await backend.close()

if __name__ == "__main__":
    print("=== Basketball Shot Analyzer Core Test ===\n")
    asyncio.run(test_ollama_backend())
