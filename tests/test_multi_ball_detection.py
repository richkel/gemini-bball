#!/usr/bin/env python
"""
Test script for multi-ball detection using both color tracking and Ollama object detection.
This script allows testing different ball types and detection methods.
"""

import argparse
import asyncio
import cv2
import logging
import numpy as np
import os
import sys
import time
from typing import Optional

# Add the src directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tracking import TrackingManager, BallType
from src.utils.ollama_detector import OllamaObjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BallDetectionTest:
    """Test class for ball detection using color tracking and Ollama."""
    
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        ball_type: str = "BASKETBALL",
        use_ollama_detection: bool = False,
        ollama_model: str = "gemma3:12b-it-q4_K_M",
        show_preview: bool = True
    ):
        """Initialize the ball detection test.
        
        Args:
            camera_index: Index of the camera to use
            width: Frame width
            height: Frame height
            ball_type: Type of ball to track
            use_ollama_detection: Whether to use Ollama for object detection
            ollama_model: Ollama model to use for object detection
            show_preview: Whether to show preview window
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.show_preview = show_preview
        
        # Convert string ball type to enum
        self.ball_type = BallType.from_string(ball_type)
        logger.info(f"Using ball type: {self.ball_type.name}")
        
        # Ollama detection settings
        self.use_ollama_detection = use_ollama_detection
        self.ollama_model = ollama_model
        if use_ollama_detection:
            logger.info(f"Using Ollama for object detection with model {ollama_model}")
        
        # Initialize components
        self.cap = None
        self.tracking_manager = None
        self.running = False
    
    async def initialize(self):
        """Initialize the test components."""
        # Initialize camera
        logger.info(f"Initializing camera {self.camera_index}")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Try to set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        # Get actual camera properties
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height}")
        
        # Initialize tracking manager
        self.tracking_manager = TrackingManager(
            enable_hand_tracking=True,
            enable_ball_tracking=True,
            ball_type=self.ball_type,
            use_ollama_detection=self.use_ollama_detection,
            ollama_model=self.ollama_model
        )
        
        return True
    
    async def run(self):
        """Run the ball detection test."""
        # Initialize components
        if not await self.initialize():
            logger.error("Initialization failed")
            return
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        # Create window if preview is enabled
        if self.show_preview:
            cv2.namedWindow("Ball Detection Test", cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    # Try to recover
                    await asyncio.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process frame with tracking manager
                processed_frame, tracking_info = self.tracking_manager.process(frame)
                
                # Add detection method info to frame
                detection_method = tracking_info.get("detection_method", "none")
                ball_detected = tracking_info.get("ball_detected", False)
                ball_type = tracking_info.get("ball_type", self.ball_type)
                
                if isinstance(ball_type, BallType):
                    ball_type_name = ball_type.name
                else:
                    ball_type_name = str(ball_type)
                
                # Add text overlay
                cv2.putText(
                    processed_frame,
                    f"Detection: {detection_method.upper()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if ball_detected else (0, 0, 255),
                    2
                )
                
                cv2.putText(
                    processed_frame,
                    f"Ball Type: {ball_type_name}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if ball_detected else (0, 0, 255),
                    2
                )
                
                # Show FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(
                    processed_frame,
                    f"FPS: {fps:.1f}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
                # Show frame if preview is enabled
                if self.show_preview:
                    cv2.imshow("Ball Detection Test", processed_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested exit")
                        break
                    elif key == ord('b'):
                        # Cycle through ball types
                        current_index = list(BallType).index(self.ball_type)
                        next_index = (current_index + 1) % len(BallType)
                        self.ball_type = list(BallType)[next_index]
                        self.tracking_manager.set_ball_type(self.ball_type)
                        logger.info(f"Switched to ball type: {self.ball_type.name}")
                    elif key == ord('o'):
                        # Toggle Ollama detection
                        self.use_ollama_detection = not self.use_ollama_detection
                        if self.tracking_manager:
                            self.tracking_manager.use_ollama_detection = self.use_ollama_detection
                        logger.info(f"Ollama detection: {'enabled' if self.use_ollama_detection else 'disabled'}")
                
                # Sleep to prevent CPU overload
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources")
        self.running = False
        
        # Release camera
        if self.cap is not None:
            self.cap.release()
        
        # Close tracking manager
        if self.tracking_manager is not None:
            self.tracking_manager.close()
        
        # Destroy windows
        if self.show_preview:
            cv2.destroyAllWindows()

async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ball Detection Test')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height (default: 720)')
    parser.add_argument('--ball-type', type=str, choices=["BASKETBALL", "TENNIS", "SOCCER", "VOLLEYBALL", "BASEBALL", "GENERIC"], 
                       default="BASKETBALL", 
                       help='Type of ball to track (default: BASKETBALL)')
    parser.add_argument('--use-ollama-detection', action='store_true',
                       help='Use Ollama for object detection')
    parser.add_argument('--ollama-model', type=str, default="gemma3:12b-it-q4_K_M",
                       help='Ollama model to use for object detection (default: gemma3:12b-it-q4_K_M)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')
    
    args = parser.parse_args()
    
    # Create and run the test
    test = BallDetectionTest(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        ball_type=args.ball_type,
        use_ollama_detection=args.use_ollama_detection,
        ollama_model=args.ollama_model,
        show_preview=not args.no_preview
    )
    
    await test.run()

if __name__ == "__main__":
    asyncio.run(main())
