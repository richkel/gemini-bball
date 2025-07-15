"""
Simple test script for Ollama backend with direct imports.
"""
import sys
from pathlib import Path
import asyncio
import cv2
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

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

async def main():
    """Main test function."""
    # Import here to avoid import issues
    from analysis.ollama_backend import OllamaBackend
    
    print("=== Basketball Shot Analyzer Simple Test ===\n")
    
    print("Initializing Ollama backend...")
    try:
        # Try with solar-pro first, fall back to llava if needed
        backend = OllamaBackend(model_name="solar-pro:latest")
        print(f"Using model: solar-pro:latest")
    except Exception as e:
        print(f"Error with solar-pro: {e}")
        print("Falling back to llava:latest")
        backend = OllamaBackend(model_name="llava:latest")
    
    print("Creating test frame...")
    test_frame = create_test_frame()
    
    print("Running analysis...")
    try:
        result = await backend.analyze_frame(test_frame, timestamp=0.0)
        if result:
            print("\n🎯 Analysis Results:")
            print(f"- Shot detected: {result.shot_type}")
            print(f"- Result: {result.result}")
            print(f"- Confidence: {result.confidence:.2f}")
            print(f"- Feedback: {result.feedback}")
        else:
            print("\nℹ️ No shot detected in the test frame.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise
    finally:
        await backend.close()
        print("\n✅ Test completed.")

if __name__ == "__main__":
    asyncio.run(main())
