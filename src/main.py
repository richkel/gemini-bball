"""
Basketball Shot Analyzer - Main Application
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import cv2
import numpy as np

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from analysis.ollama_backend import OllamaBackend
from models import ShotAnalysis, ShotStatistics, ShotResult, ShotType
from utils.frame_processor import preprocess_frame, draw_shot_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BasketballShotAnalyzer:
    """Main application class for basketball shot analysis."""
    
    def __init__(self, backend: str = "ollama", model_name: str = "llava:latest"):
        """Initialize the analyzer with the specified backend.
        
        Args:
            backend: Analysis backend to use ('ollama' or 'gemini')
            model_name: Name of the model to use with the backend
        """
        self.backend_type = backend.lower()
        self.model_name = model_name
        self.backend = self._initialize_backend()
        self.last_analysis: Optional[ShotAnalysis] = None
        self.stats = ShotStatistics()
        
    def _initialize_backend(self):
        """Initialize the analysis backend."""
        if self.backend_type == "ollama":
            return OllamaBackend(model_name=self.model_name)
        elif self.backend_type == "gemini":
            # We'll implement the Gemini backend later
            raise NotImplementedError("Gemini backend not yet implemented")
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")
    
    async def process_video(
        self, 
        video_path: Union[str, Path], 
        output_path: Optional[Union[str, Path]] = None,
        show_preview: bool = True
    ) -> None:
        """Process a video file for shot analysis.
        
        Args:
            video_path: Path to the input video file
            output_path: Optional path to save the output video
            show_preview: Whether to show a live preview
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), 
                fourcc, 
                fps, 
                (width, height)
            )
        
        try:
            frame_idx = 0
            shot_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                timestamp = frame_idx / fps
                
                # Process every 30th frame (adjust based on needs)
                if frame_idx % 30 == 0:
                    # Make a copy of the frame for analysis
                    analysis_frame = frame.copy()
                    
                    # Run analysis in the background
                    shot_analysis = await self.backend.analyze_frame(
                        frame=analysis_frame,
                        timestamp=timestamp
                    )
                    
                    if shot_analysis:
                        self.last_analysis = shot_analysis
                        self.stats = self.backend.stats
                        shot_frames = 0
                
                # Draw analysis results if available
                if self.last_analysis and shot_frames < 90:  # Show for 3 seconds at 30 FPS
                    frame = draw_shot_result(
                        frame=frame,
                        shot_analysis=self.last_analysis,
                        stats=self.stats
                    )
                    shot_frames += 1
                
                # Write frame to output video
                if writer:
                    writer.write(frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Basketball Shot Analyzer', frame)
                    
                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            logger.info("Video processing complete")
            
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            await self.backend.close()

async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Basketball Shot Analyzer')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output', '-o', type=str, help='Path to save the output video')
    parser.add_argument('--backend', type=str, default='ollama', 
                        choices=['ollama', 'gemini'], 
                        help='Analysis backend to use')
    parser.add_argument('--model', type=str, default='llava:latest',
                       help='Model to use with the backend')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable the preview window')
    
    args = parser.parse_args()
    
    try:
        analyzer = BasketballShotAnalyzer(backend=args.backend, model_name=args.model)
        await analyzer.process_video(
            video_path=args.video_path,
            output_path=args.output,
            show_preview=not args.no_preview
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
