"""
Real-time Basketball Shot Analyzer

This module provides a real-time video processing pipeline for analyzing
basketball shots using either the Ollama or Gemini backend.
"""

import asyncio
import cv2
import time
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

from ..analysis.base import AnalysisBackend
from ..models.shot_models import ShotAnalysis, ShotStatistics, ShotType, ShotResult
from ..utils.tracking import TrackingManager, BallType

logger = logging.getLogger(__name__)

class BackendType(Enum):
    """Enum for different analysis backends."""
    OLLAMA = auto()
    GEMINI = auto()

@dataclass
class PipelineConfig:
    """Configuration for the real-time pipeline."""
    backend_type: BackendType = BackendType.OLLAMA
    model_name: str = "llava:latest"
    video_source: Any = 0  # Can be camera index or video file path
    frame_skip: int = 10  # Process every nth frame (increased for better performance)
    frame_width: int = 640  # Frame width for processing
    frame_height: int = 480  # Frame height for processing
    show_preview: bool = True
    save_output: bool = False
    output_file: str = "output.mp4"
    api_key: Optional[str] = None  # Required for Gemini
    enable_hand_tracking: bool = True
    enable_ball_tracking: bool = True
    ball_type: Any = None  # Type of ball to track
    use_ollama_detection: bool = False  # Whether to use Ollama for object detection
    ollama_model: str = "gemma3:12b-it-q4_K_M"  # Ollama model for object detection
    low_vram_mode: bool = True  # Optimizations for systems with <10GB VRAM
    enable_caching: bool = True  # Enable frame caching to avoid reprocessing
    ollama_timeout: float = 30.0  # Timeout for Ollama requests
    ollama_retry_count: int = 2  # Number of retries for failed Ollama requests
    ollama_retry_delay: float = 1.0  # Delay between retries in seconds

class RealTimeAnalyzer:
    """Real-time basketball shot analyzer."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the real-time analyzer."""
        self.config = config
        self.backend: Optional[AnalysisBackend] = None
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.last_analysis: Optional[ShotAnalysis] = None
        self.analysis_history: list[ShotAnalysis] = []
        self.stats = ShotStatistics()
        self.tracking_manager = None
        self._frame_cache = {}  # Cache for frame analysis results
        
        # Initialize tracking manager if enabled
        if config.enable_hand_tracking or config.enable_ball_tracking:
            self.tracking_manager = TrackingManager(
                enable_hand_tracking=config.enable_hand_tracking,
                enable_ball_tracking=config.enable_ball_tracking,
                ball_type=config.ball_type if config.ball_type is not None else BallType.BASKETBALL,
                use_ollama_detection=config.use_ollama_detection,
                ollama_model=config.ollama_model,
                low_vram_mode=config.low_vram_mode
            )
        
        logger.info(f"Initialized RealTimeAnalyzer with {config.backend_type.name} backend")
        if config.low_vram_mode:
            logger.info("Low VRAM mode enabled - using optimized settings for systems with <10GB VRAM")
    
    async def initialize(self):
        """Asynchronously initialize the analyzer.
        
        This method is called by WebcamAnalyzer to initialize the backend
        and prepare for analysis.
        """
        # Initialize the backend
        self._init_backend()
        
        # Any other async initialization can go here
        if self.backend:
            # Some backends might need async initialization
            if hasattr(self.backend, 'initialize') and callable(getattr(self.backend, 'initialize')):
                await self.backend.initialize()
                
    async def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return the frame with overlays.
        
        This method is called by WebcamAnalyzer to process each frame.
        
        Args:
            frame: Input frame as a numpy array (RGB format)
            
        Returns:
            Processed frame with overlays
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Process the frame
        analysis, processed_frame, tracking_results = await self._process_frame(frame)
        
        # Draw overlays on the frame
        result_frame = self._draw_overlay(processed_frame, analysis, tracking_results)
        
        return result_frame
    
    def _init_backend(self):
        """Initialize the analysis backend based on config."""
        if self.config.backend_type == BackendType.OLLAMA:
            from ..analysis.ollama_backend import OllamaBackend
            self.backend = OllamaBackend(model_name=self.config.model_name)
        elif self.config.backend_type == BackendType.GEMINI:
            from ..analysis.gemini_backend import GeminiBackend
            if not self.config.api_key:
                raise ValueError("API key is required for Gemini backend")
            self.backend = GeminiBackend(
                api_key=self.config.api_key,
                model_name=self.config.model_name
            )
        else:
            raise ValueError(f"Unsupported backend type: {self.config.backend_type}")
    
    async def _frame_hash(self, frame: np.ndarray) -> int:
        """Create a simple hash of downsampled frame for caching.
        
        Args:
            frame: Input frame as a numpy array
            
        Returns:
            Integer hash of the frame
        """
        # Downsample frame to tiny resolution for faster hashing
        tiny = cv2.resize(frame, (16, 16))
        # Convert to grayscale to further reduce dimensionality
        gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
        # Create a hash from the bytes
        return hash(gray.tobytes())
    
    async def _process_frame(self, frame: np.ndarray) -> Tuple[Optional[ShotAnalysis], np.ndarray, Dict[str, Any]]:
        """Process a single frame for shot analysis.
        
        Args:
            frame: Input frame as a numpy array (BGR format)
            
        Returns:
            Tuple of (shot analysis, processed frame with tracking overlays, tracking results)
        """
        # Resize frame to smaller resolution for faster processing
        # Original frame is preserved for display if needed
        process_frame = cv2.resize(frame, (640, 480))
        
        # Process tracking if enabled
        tracking_results = {}
        if self.tracking_manager is not None:
            annotated_frame, tracking_results = self.tracking_manager.process(process_frame)
        else:
            annotated_frame = process_frame.copy()
        
        # Only analyze every nth frame based on frame_skip
        analysis = None
        if self.frame_count % self.config.frame_skip == 0 and self.backend is not None:
            # Check frame cache to avoid reprocessing similar frames
            frame_hash = await self._frame_hash(process_frame)
            
            # Use cache if available (within last 30 frames)
            cache_hit = False
            if hasattr(self, '_frame_cache') and frame_hash in self._frame_cache:
                cached_result, cached_frame = self._frame_cache[frame_hash]
                # Only use cache if it's recent (within last 5 seconds)
                if time.time() - cached_result.timestamp < 5.0:
                    analysis = cached_result
                    cache_hit = True
                    logger.debug(f"Cache hit for frame {self.frame_count}")
            
            # If not cached, send the frame to the backend for analysis
            if not cache_hit:
                timestamp = time.time()
                analysis = None
                retry_count = 0
                
                while retry_count < self.config.ollama_retry_count:
                    try:
                        analysis = await self.backend.analyze_frame(process_frame, timestamp)
                        break  # Success, break out of retry loop
                    except httpx.TimeoutException:
                        logger.warning(f"Ollama request timed out (attempt {retry_count + 1}/{self.config.ollama_retry_count})")
                        if retry_count < self.config.ollama_retry_count - 1:
                            await asyncio.sleep(self.config.ollama_retry_delay)
                    except Exception as e:
                        logger.error(f"Error analyzing frame: {e}")
                        break
                    
                    retry_count += 1
                
                # If a shot is detected, update the last analysis
                if analysis:
                    self.last_analysis = analysis
                    self.analysis_history.append(analysis)
                    self.stats.update(analysis)
                    logger.info(f"Shot detected: {analysis.shot_type.name} - {analysis.result.name}")
                else:
                    logger.debug("No shot detected in frame")
                    
                # Cache the result
                if self.config.enable_caching:
                    self._frame_cache[frame_hash] = (analysis, process_frame)
        
        return analysis, process_frame, tracking_results
    
    def _draw_overlay(self, frame: np.ndarray, analysis: Optional[ShotAnalysis], tracking_results: Dict[str, Any]):
        """Draw analysis results and tracking information on the frame."""
        # Create a clean overlay surface
        overlay = np.zeros_like(frame)
        
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)
        thickness = 1
        
        # Add shot statistics
        stats_text = (f"Shots: {self.stats.total_shots} | "
                     f"Made: {self.stats.made_shots} | "
                     f"FG%: {self.stats.field_goal_percentage:.1f}%")
        cv2.putText(overlay, stats_text, (10, 60), font, 0.6, color, thickness)
        
        # Add last shot info if available
        if analysis:
            shot_text = (f"Last Shot: {analysis.shot_type.name.replace('_', ' ').title()}"
                       f" - {analysis.result.name.upper()}")
            cv2.putText(overlay, shot_text, (10, 90), font, 0.6, color, thickness)
            
            # Add feedback
            feedback_text = f"Feedback: {analysis.feedback[:50]}..."
            cv2.putText(overlay, feedback_text, (10, 120), font, 0.5, color, thickness)
        
        # Add tracking status information
        y_offset = 150
        
        # Add hand tracking status
        if tracking_results.get("hand_tracking"):
            hand_landmarks = tracking_results["hand_tracking"].get("multi_hand_landmarks")
            if hand_landmarks:
                num_hands = len(hand_landmarks)
                cv2.putText(overlay, f"Hands Detected: {num_hands}", (10, y_offset),
                           font, 0.5, (0, 255, 255), thickness)
                y_offset += 30
        
        # Add ball tracking status
        if tracking_results.get("ball_tracking"):
            ball_detected = tracking_results["ball_tracking"].get("ball_detected", False)
            if ball_detected:
                ball_center = tracking_results["ball_tracking"].get("ball_center")
                if ball_center:
                    cv2.putText(overlay, f"Ball Position: ({ball_center[0]}, {ball_center[1]})", (10, y_offset),
                               font, 0.5, (0, 255, 255), thickness)
                    y_offset += 30
        
        # Add backend info
        backend_text = f"Backend: {self.config.backend_type.name}"
        cv2.putText(overlay, backend_text, (10, frame.shape[0] - 20),
                   font, 0.5, color, thickness)
        
        # Combine overlay with original frame
        result_frame = cv2.addWeighted(frame, 1.0, overlay, 0.75, 0)
        
        return result_frame
    
    async def run(self):
        """Run the real-time analysis pipeline."""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.config.video_source)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open video source")
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Initialize video writer if saving output
            if self.config.save_output:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(
                    self.config.output_file,
                    fourcc,
                    fps,
                    (width, height)
                )
            
            logger.info(f"Starting real-time analysis (Press 'q' to quit)")
            
            while True:
                # Read a frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Increment frame counter
                self.frame_count += 1
                
                # Process the frame (includes tracking and analysis)
                analysis, processed_frame, tracking_results = await self._process_frame(frame)
                
                # Draw overlay
                overlay = self._draw_overlay(processed_frame, analysis or self.last_analysis, tracking_results)
                
                # Display the frame
                if self.config.show_preview:
                    cv2.imshow('Basketball Shot Analysis', overlay)
                    
                # Save the frame
                if self.config.save_output and self.writer is not None:
                    self.writer.write(overlay)
                    
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to quit")
                    break
                
                # Add a small delay to prevent high CPU usage
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in real-time pipeline: {e}", exc_info=True)
            
        finally:
            # Clean up
            if self.cap is not None:
                self.cap.release()
                
            if self.writer is not None:
                self.writer.release()
                
            if self.tracking_manager is not None:
                self.tracking_manager.close()
                
            cv2.destroyAllWindows()
            if self.backend:
                await self.backend.close()
            
            logger.info("Real-time analysis completed")
            
            # Print final statistics
            print("\n📊 Final Statistics:")
            print(f"- Total Frames Processed: {self.frame_count}")
            print(f"- Total Shots Detected: {self.stats.total_shots}")
            print(f"- Shots Made: {self.stats.made_shots}")
            print(f"- Field Goal %: {self.stats.field_goal_percentage:.1f}%")
            
            # Save analysis results
            self._save_analysis_results()
    
    def cleanup(self):
        """Synchronous cleanup method for non-async contexts.
        
        This method is called when the analyzer is being cleaned up
        from a non-async context, such as when the WebcamAnalyzer
        is being cleaned up.
        """
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            
        # Release video writer
        if self.writer is not None:
            self.writer.release()
            
        # Close tracking manager
        if self.tracking_manager is not None:
            self.tracking_manager.close()
            
        # Destroy windows
        cv2.destroyAllWindows()
        
        # Note: We can't close the backend here because it requires
        # an async context. The caller should handle that separately
        # if needed.
        self._save_analysis_results()
    
    def _save_analysis_results(self):
        """Save the analysis results to a JSON file."""
        import json
        from datetime import datetime
        
        if not self.analysis_history:
            logger.info("No analysis results to save")
            return
        
        # Prepare results
        results = {
            "timestamp": datetime.now().isoformat(),
            "backend": self.config.backend_type.name,
            "model": self.config.model_name,
            "total_frames": self.frame_count,
            "total_shots": self.stats.total_shots,
            "made_shots": self.stats.made_shots,
            "missed_shots": self.stats.missed_shots,
            "field_goal_percentage": self.stats.field_goal_percentage,
            "shots": [
                {
                    "timestamp": shot.timestamp,
                    "shot_type": shot.shot_type.name,
                    "result": shot.result.name,
                    "confidence": shot.confidence,
                    "feedback": shot.feedback
                }
                for shot in self.analysis_history
            ]
        }
        
        # Save to file
        filename = f"shot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {filename}")

async def main():
    """Main function to run the real-time analyzer."""
    import argparse
    from ..utils.tracking import BallType
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Basketball Shot Analyzer')
    parser.add_argument('--backend', type=str, default='ollama',
                       choices=['ollama', 'gemini'],
                       help='Backend to use for analysis (default: ollama)')
    parser.add_argument('--model', type=str, default='llava:latest',
                       help='Model to use for analysis (default: llava:latest)')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera index or video file path, default: 0)')
    parser.add_argument('--frame-skip', type=int, default=10,
                       help='Process every N frames (default: 10)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')
    parser.add_argument('--save', action='store_true',
                       help='Save output to file')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Output file path (default: output.mp4)')
    parser.add_argument('--api-key', type=str,
                       help='API key (required for Gemini backend)')
    parser.add_argument('--ball-type', type=str, choices=["BASKETBALL", "TENNIS", "SOCCER", "VOLLEYBALL", "BASEBALL", "GENERIC"], 
                       default="BASKETBALL", 
                       help='Type of ball to track (default: BASKETBALL)')
    parser.add_argument('--use-ollama-detection', action='store_true',
                       help='Use Ollama for object detection as fallback')
    parser.add_argument('--ollama-model', type=str, default="gemma3:12b-it-q4_K_M",
                       help='Ollama model to use for object detection (default: gemma3:12b-it-q4_K_M)')
    parser.add_argument('--low-vram', action='store_true',
                       help='Enable optimizations for systems with <10GB VRAM')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable frame caching')
    parser.add_argument('--frame-width', type=int, default=640,
                       help='Frame width for processing (default: 640)')
    parser.add_argument('--frame-height', type=int, default=480,
                       help='Frame height for processing (default: 480)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = PipelineConfig(
        backend_type=BackendType.OLLAMA if args.backend == 'ollama' else BackendType.GEMINI,
        model_name=args.model,
        video_source=args.source if args.source.isalpha() or '.' in args.source else int(args.source),
        frame_skip=args.frame_skip,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        show_preview=not args.no_preview,
        save_output=args.save,
        output_file=args.output,
        api_key=args.api_key or os.getenv('GOOGLE_API_KEY'),
        ball_type=BallType.from_string(args.ball_type) if hasattr(args, 'ball_type') else BallType.BASKETBALL,
        use_ollama_detection=args.use_ollama_detection if hasattr(args, 'use_ollama_detection') else False,
        ollama_model=args.ollama_model if hasattr(args, 'ollama_model') else "gemma3:12b-it-q4_K_M",
        low_vram_mode=args.low_vram if hasattr(args, 'low_vram') else True,
        enable_caching=not args.no_cache if hasattr(args, 'no_cache') else True
    )
    
    # Create and run the analyzer
    analyzer = RealTimeAnalyzer(config)
    await analyzer.run()

if __name__ == "__main__":
    import os
    asyncio.run(main())
