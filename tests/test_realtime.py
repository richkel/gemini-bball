"""
Test script for the real-time basketball shot analyzer.

This script tests the real-time analysis pipeline with a test video or camera feed.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the real-time analyzer
from src.pipeline.real_time_analyzer import RealTimeAnalyzer, PipelineConfig, BackendType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run the real-time analyzer with test settings."""
    print("=== Basketball Shot Analyzer - Real-time Test ===\n")
    
    # Configuration
    config = PipelineConfig(
        backend_type=BackendType.OLLAMA,
        model_name="llava:latest",
        video_source=0,  # Use webcam (change to video file path if desired)
        frame_skip=10,   # Process every 10th frame for better performance
        show_preview=True,
        save_output=True,
        output_file="output.mp4"
    )
    
    # Create and run the analyzer
    analyzer = RealTimeAnalyzer(config)
    
    print("Starting real-time analysis...")
    print("Press 'q' to quit the preview window.")
    
    try:
        await analyzer.run()
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user.")
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
    finally:
        print("\n✅ Test completed.")
        
        # Print final statistics if available
        if hasattr(analyzer, 'stats'):
            print("\n📊 Final Statistics:")
            print(f"- Total Frames Processed: {analyzer.frame_count}")
            print(f"- Total Shots Detected: {analyzer.stats.total_shots}")
            print(f"- Shots Made: {analyzer.stats.made_shots}")
            print(f"- Field Goal %: {analyzer.stats.field_goal_percentage:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
