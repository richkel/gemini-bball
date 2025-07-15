"""
Basketball Shot Analyzer - Webcam Integration

This script runs the basketball shot analyzer with a webcam feed.
"""
import cv2
import asyncio
import logging
import time
from typing import Optional, Dict, Any
import numpy as np
import argparse

# Import BackendType from the real_time_analyzer module
from src.pipeline.real_time_analyzer import BackendType, PipelineConfig, RealTimeAnalyzer
from src.utils.tracking import BallType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebcamAnalyzer:
    """Handles webcam integration with the basketball shot analyzer."""
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend_type: str = "ollama",
        model_name: str = "gemma3:12b-it-q4_K_M",
        api_key: Optional[str] = None,
        frame_skip: int = 5,
        enable_hand_tracking: bool = True,
        enable_ball_tracking: bool = True,
        ball_type: str = "BASKETBALL",
        use_ollama_detection: bool = True,
        ollama_model: str = "gemma3:12b-it-q4_K_M"
    ):
        """Initialize the webcam analyzer.
        
        Args:
            camera_index: Index of the camera to use
            width: Frame width
            height: Frame height
            fps: Target frames per second
            backend_type: Backend to use ('ollama' or 'gemini')
            model_name: Model to use for analysis
            api_key: API key (required for Gemini backend)
            frame_skip: Process every N frames
            enable_hand_tracking: Whether to enable hand tracking
            enable_ball_tracking: Whether to enable ball tracking
            ball_type: Type of ball to track (BASKETBALL, TENNIS, SOCCER, VOLLEYBALL, BASEBALL, GENERIC)
            use_ollama_detection: Whether to use Ollama for object detection
            ollama_model: Ollama model to use for object detection
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_skip = frame_skip
        self.model_name = model_name  # Store model_name as instance variable
        self.api_key = api_key  # Store API key as instance variable
        self.enable_hand_tracking = enable_hand_tracking  # Store hand tracking setting
        self.enable_ball_tracking = enable_ball_tracking  # Store ball tracking setting
        
        # Convert string ball type to enum
        self.ball_type = BallType.from_string(ball_type)
        logger.info(f"Using ball type: {self.ball_type.name}")
        
        # Ollama detection settings
        self.use_ollama_detection = use_ollama_detection
        self.ollama_model = ollama_model
        if use_ollama_detection:
            logger.info(f"Using Ollama for object detection with model {ollama_model}")
        
        # Initialize the analyzer
        self.analyzer = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        
        # Set the backend type
        if backend_type.lower() == "ollama":
            self.backend_type = BackendType.OLLAMA
        elif backend_type.lower() == "gemini":
            self.backend_type = BackendType.GEMINI
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
            
        # Set up pipeline configuration
        self.config = PipelineConfig(
            backend_type=self.backend_type,
            model_name=model_name,
            video_source=self.camera_index,
            frame_skip=self.frame_skip,
            show_preview=True,
            save_output=False,
            api_key=api_key,
            enable_hand_tracking=enable_hand_tracking,
            enable_ball_tracking=enable_ball_tracking,
            ball_type=self.ball_type,
            use_ollama_detection=use_ollama_detection,
            ollama_model=ollama_model
        )
    
    async def initialize(self):
        """Initialize the webcam and analyzer."""
        logger.info("Initializing webcam...")
        
        # Try multiple times to open the camera
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Release any existing capture
                if self.cap is not None:
                    self.cap.release()
                    
                # Create a new capture
                self.cap = cv2.VideoCapture(self.camera_index)
                
                if not self.cap.isOpened():
                    if attempt < max_attempts - 1:
                        logger.warning(f"Failed to open camera on attempt {attempt+1}/{max_attempts}, retrying...")
                        await asyncio.sleep(1)  # Wait before retrying
                        continue
                    else:
                        raise RuntimeError(f"Could not open camera at index {self.camera_index} after {max_attempts} attempts")
                
                # Camera opened successfully
                break
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Error initializing camera on attempt {attempt+1}/{max_attempts}: {e}, retrying...")
                    await asyncio.sleep(1)  # Wait before retrying
                else:
                    raise RuntimeError(f"Failed to initialize camera after {max_attempts} attempts: {e}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Verify camera is working by reading a test frame
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            raise RuntimeError("Camera opened but failed to capture test frame")
        
        logger.info(f"Webcam initialized: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS")
        
        # Initialize the analyzer
        from src.pipeline.real_time_analyzer import RealTimeAnalyzer, PipelineConfig, BackendType
        
        config = PipelineConfig(
            backend_type=self.backend_type,
            model_name=self.model_name,
            video_source=self.camera_index,
            frame_skip=self.frame_skip,
            show_preview=True,
            save_output=False,
            api_key=self.api_key,
            enable_hand_tracking=self.enable_hand_tracking,
            enable_ball_tracking=self.enable_ball_tracking
        )
        
        self.analyzer = RealTimeAnalyzer(config)
        await self.analyzer.initialize()
        logger.info("Analyzer initialized")
    
    async def process_frame(self, frame):
        """Process a single frame through the analyzer."""
        # Convert BGR to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame through the analyzer
        processed_frame = await self.analyzer.process_frame(rgb_frame)
        
        # Convert back to BGR for display
        return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
    
    async def run(self):
        """Run the main analysis loop."""
        if not self.cap or not self.cap.isOpened():
            await self.initialize()
        
        self.running = True
        logger.info("Starting analysis. Press 'q' to quit.")
        
        try:
            while self.running:
                # Read frame with retry mechanism
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        break  # Successfully captured frame
                    
                    retry_count += 1
                    logger.warning(f"Failed to capture frame, retry {retry_count}/{max_retries}")
                    await asyncio.sleep(0.1)  # Short delay before retry
                    
                    # Try to reinitialize camera if all retries fail
                    if retry_count == max_retries - 1:
                        logger.warning("Attempting to reinitialize camera...")
                        try:
                            if self.cap is not None:
                                self.cap.release()
                            self.cap = cv2.VideoCapture(self.camera_index)
                            if not self.cap.isOpened():
                                logger.error("Failed to reinitialize camera")
                                break
                        except Exception as e:
                            logger.error(f"Error reinitializing camera: {e}")
                
                # If we still couldn't get a frame after retries
                if not ret or frame is None:
                    logger.error("Failed to capture frame after multiple retries")
                    break
                
                # Process frame
                try:
                    processed_frame = await self.process_frame(frame)
                    
                    # Display the processed frame
                    cv2.imshow('Basketball Shot Analyzer', processed_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit signal received")
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing frame: {e}", exc_info=True)
                
                # Small delay to control frame rate
                await asyncio.sleep(1.0 / self.fps)
                
        except KeyboardInterrupt:
            logger.info("Analysis stopped by user")
        except Exception as e:
            logger.error(f"Error in analysis loop: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'analyzer') and self.analyzer:
            # Check if the analyzer has a cleanup method
            if hasattr(self.analyzer, 'cleanup') and callable(getattr(self.analyzer, 'cleanup')):
                self.analyzer.cleanup()
        logger.info("Resources cleaned up")

async def main():
    """Main function to run the webcam analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Basketball Shot Analyzer with Webcam')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS (default: 30)')
    parser.add_argument('--backend', type=str, default='ollama',
                       choices=['ollama', 'gemini'],
                       help='Backend to use (default: ollama)')
    parser.add_argument('--model', type=str, default='gemma3:12b-it-q4_K_M',
                       help='Model to use for analysis (default: gemma3:12b-it-q4_K_M)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key (required for Gemini backend)')
    parser.add_argument('--frame-skip', type=int, default=5,
                       help='Process every N frames (default: 5)')
    parser.add_argument('--no-hand-tracking', action='store_true',
                       help='Disable hand tracking')
    parser.add_argument('--no-ball-tracking', action='store_true',
                       help='Disable ball tracking')
    parser.add_argument('--ball-type', type=str, choices=["BASKETBALL", "TENNIS", "SOCCER", "VOLLEYBALL", "BASEBALL", "GENERIC"], 
                       default="BASKETBALL", 
                       help='Type of ball to track (default: BASKETBALL)')
    parser.add_argument('--use-ollama-detection', action='store_true',
                       help='Use Ollama for object detection as fallback')
    parser.add_argument('--ollama-model', type=str, default="gemma3:12b-it-q4_K_M",
                       help='Ollama model to use for object detection (default: gemma3:12b-it-q4_K_M)')
    
    args = parser.parse_args()
    
    # Create and run the analyzer
    analyzer = WebcamAnalyzer(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend_type=args.backend,
        model_name=args.model,
        api_key=args.api_key,
        frame_skip=args.frame_skip,
        enable_hand_tracking=not args.no_hand_tracking,
        enable_ball_tracking=not args.no_ball_tracking,
        ball_type=args.ball_type,
        use_ollama_detection=args.use_ollama_detection,
        ollama_model=args.ollama_model
    )
    
    # Print configuration
    logger.info(f"Starting webcam analyzer with the following configuration:")
    logger.info(f"  Camera: {args.camera}")
    logger.info(f"  Resolution: {args.width}x{args.height}")
    logger.info(f"  Backend: {args.backend}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Ball type: {args.ball_type}")
    logger.info(f"  Hand tracking: {not args.no_hand_tracking}")
    logger.info(f"  Ball tracking: {not args.no_ball_tracking}")
    
    try:
        await analyzer.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
