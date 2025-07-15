"""
Example script for using the Basketball Shot Analyzer with Ollama.

This script demonstrates how to use the BasketballShotAnalyzer class to analyze
a basketball video using the Ollama backend.

Usage:
    python example.py path/to/your/video.mp4
"""
import asyncio
import logging
import os
from pathlib import Path
import sys

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from main import BasketballShotAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    if len(sys.argv) < 2:
        print("Usage: python example.py path/to/your/video.mp4")
        return
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return 1
        
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / f"analyzed_{Path(video_path).name}")
    
    try:
        print("Available models:")
        models = ["llava:latest", "llama3:latest"]  # Add other models you have
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
            
        model_choice = input(f"\nSelect model (1-{len(models)}), or press Enter for default (llava:latest): ")
        model_name = models[int(model_choice) - 1] if model_choice.strip() else "llava:latest"
        
        print(f"\nInitializing analyzer with {model_name}...")
        analyzer = BasketballShotAnalyzer(
            backend="ollama",
            model_name=model_name
        )
        
        print(f"Starting analysis of {video_path}...")
        print("Press 'q' to quit the preview window.")
        
        # Process the video
        await analyzer.process_video(
            video_path=video_path,
            output_path=output_path,
            show_preview=True
        )
        
        print(f"Analysis complete! Output saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())
