#!/usr/bin/env python
"""
Test script for the OllamaObjectDetector's detect_ball method.
This script loads an image and uses the OllamaObjectDetector to detect balls.
"""

import argparse
import cv2
import logging
import numpy as np
import os
import sys
import time
from typing import Optional

# Add the src directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.ollama_detector import OllamaObjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ollama_detector(
    image_path: str,
    model_name: str = "gemma3:12b-it-q4_K_M",
    server_url: str = "http://localhost:11434",
    timeout: int = 30,
    max_retries: int = 3
):
    """Test the OllamaObjectDetector's detect_ball method.
    
    Args:
        image_path: Path to the image to test
        model_name: Ollama model to use
        server_url: Ollama server URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
    """
    # Load the image
    logger.info(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image from {image_path}")
        return
    
    # Create the detector
    logger.info(f"Creating OllamaObjectDetector with model {model_name}")
    detector = OllamaObjectDetector(
        model_name=model_name,
        server_url=server_url,
        timeout=timeout,
        max_retries=max_retries
    )
    
    # Detect balls
    logger.info("Detecting balls...")
    start_time = time.time()
    result = detector.detect_ball(image)
    elapsed_time = time.time() - start_time
    
    # Log the result
    logger.info(f"Detection completed in {elapsed_time:.2f} seconds")
    logger.info(f"Result: {result}")
    
    # Visualize the result
    if result["detected"] and result["coordinates"]:
        # Draw bounding box
        x1, y1, x2, y2 = result["coordinates"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw ball type and confidence
        ball_type = result.get("ball_type", "unknown")
        confidence = result.get("confidence", 0.0)
        cv2.putText(
            image,
            f"{ball_type} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
        
        # Show the image
        cv2.imshow("Ball Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test OllamaObjectDetector')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the image to test')
    parser.add_argument('--model', type=str, default="gemma3:12b-it-q4_K_M",
                       help='Ollama model to use (default: gemma3:12b-it-q4_K_M)')
    parser.add_argument('--server', type=str, default="http://localhost:11434",
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--retries', type=int, default=3,
                       help='Maximum number of retries (default: 3)')
    
    args = parser.parse_args()
    
    # Run the test
    test_ollama_detector(
        image_path=args.image,
        model_name=args.model,
        server_url=args.server,
        timeout=args.timeout,
        max_retries=args.retries
    )

if __name__ == "__main__":
    main()
