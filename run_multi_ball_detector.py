#!/usr/bin/env python
"""
Multi-ball detection and tracking demo script.

This script demonstrates the multi-ball detection and tracking capabilities
of the basketball analyzer using both color-based tracking and Ollama-based
object detection.
"""

import argparse
import cv2
import logging
import os
import sys
import time
from typing import Optional

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.tracking import TrackingManager, BallType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_webcam_detector(
    ball_type: BallType = BallType.BASKETBALL,
    use_ollama: bool = False,
    ollama_model: str = "gemma3:12b-it-q4_K_M",
    camera_id: int = 0,
    width: int = 640,
    height: int = 480
):
    """Run the multi-ball detector on a webcam feed.
    
    Args:
        ball_type: Type of ball to track
        use_ollama: Whether to use Ollama for object detection
        ollama_model: Ollama model to use
        camera_id: Camera ID to use
        width: Frame width
        height: Frame height
    """
    # Initialize the tracking manager
    tracking_manager = TrackingManager(
        enable_hand_tracking=True,
        enable_ball_tracking=True,
        ball_type=ball_type,
        use_ollama_detection=use_ollama,
        ollama_model=ollama_model
    )
    
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
        return
    
    logger.info(f"Starting multi-ball detector with ball_type={ball_type.name}, use_ollama={use_ollama}")
    
    # Create window
    window_name = "Multi-Ball Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Process frame
            start_time = time.time()
            annotated_frame, tracking_info = tracking_manager.process(frame)
            processing_time = time.time() - start_time
            
            # Add FPS counter and detection method
            fps = 1.0 / processing_time if processing_time > 0 else 0
            detection_method = tracking_info.get("detection_method", "none")
            ball_detected = tracking_info.get("ball_detected", False)
            ball_type_detected = tracking_info.get("ball_type", ball_type)
            
            # Format ball type for display
            if isinstance(ball_type_detected, BallType):
                ball_type_name = ball_type_detected.name
            else:
                ball_type_name = str(ball_type_detected)
            
            # Add info to frame
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f} | Method: {detection_method} | Ball: {ball_type_name} | Detected: {ball_detected}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow(window_name, annotated_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                # Quit
                break
            elif key == ord('b'):
                # Cycle through ball types
                ball_types = list(BallType)
                current_idx = ball_types.index(ball_type)
                next_idx = (current_idx + 1) % len(ball_types)
                ball_type = ball_types[next_idx]
                tracking_manager.set_ball_type(ball_type)
                logger.info(f"Switched to ball type: {ball_type.name}")
            elif key == ord('o'):
                # Toggle Ollama detection
                use_ollama = not use_ollama
                tracking_manager.use_ollama_detection = use_ollama
                logger.info(f"Ollama detection: {'enabled' if use_ollama else 'disabled'}")
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"ball_detection_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Saved frame to {filename}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Clean up
        logger.info("Cleaning up resources")
        tracking_manager.close()
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-ball detection and tracking demo')
    
    # Ball type selection
    parser.add_argument('--ball-type', type=str, choices=[bt.name for bt in BallType], 
                        default='BASKETBALL', help='Type of ball to track')
    
    # Ollama options
    parser.add_argument('--use-ollama', action='store_true', 
                        help='Use Ollama for object detection')
    parser.add_argument('--ollama-model', type=str, default='gemma3:12b-it-q4_K_M',
                        help='Ollama model to use')
    
    # Camera options
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera ID to use')
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height')
    
    args = parser.parse_args()
    
    # Convert ball type string to enum
    ball_type = BallType[args.ball_type]
    
    # Run the detector
    run_webcam_detector(
        ball_type=ball_type,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        camera_id=args.camera_id,
        width=args.width,
        height=args.height
    )

if __name__ == "__main__":
    main()
