#!/usr/bin/env python
"""
Test script for the OllamaObjectDetector's detect_ball method using a URL image.
"""

import argparse
import cv2
import logging
import numpy as np
import os
import sys
import time
import requests
from typing import Optional
from io import BytesIO

# Add the src directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.ollama_detector import OllamaObjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_image_from_url(url):
    """Download an image from a URL."""
    try:
        # Set a proper User-Agent header
        headers = {
            'User-Agent': 'GeminiBballProject/1.0 (https://github.com/user/gemini-bball; user@example.com) Python/3.x requests/2.x'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Convert to numpy array for OpenCV
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Failed to decode image from URL: {url}")
            return None
            
        return image
    except Exception as e:
        logger.error(f"Error downloading image from URL: {e}")
        return None

def test_ollama_detector(
    image_url: str,
    model_name: str = "gemma3:12b-it-q4_K_M",
    server_url: str = "http://localhost:11434",
    timeout: int = 30,
    max_retries: int = 3,
    save_image: bool = True
):
    """Test the OllamaObjectDetector's detect_ball method with a URL image.
    
    Args:
        image_url: URL of the image to test
        model_name: Ollama model to use
        server_url: Ollama server URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        save_image: Whether to save the annotated image
    """
    # Download the image
    logger.info(f"Downloading image from {image_url}")
    image = download_image_from_url(image_url)
    if image is None:
        return
    
    # Save original image
    if save_image:
        cv2.imwrite("test_images/original_image.jpg", image)
        logger.info("Saved original image to test_images/original_image.jpg")
    
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
        
        # Save the annotated image
        if save_image:
            cv2.imwrite("test_images/annotated_image.jpg", image)
            logger.info("Saved annotated image to test_images/annotated_image.jpg")
        
        # Show the image
        cv2.imshow("Ball Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test OllamaObjectDetector with URL image')
    parser.add_argument('--url', type=str, default="https://m.media-amazon.com/images/I/61STrTHHI-L._AC_SL1000_.jpg",
                       help='URL of the image to test')
    parser.add_argument('--model', type=str, default="gemma3:12b-it-q4_K_M",
                       help='Ollama model to use (default: gemma3:12b-it-q4_K_M)')
    parser.add_argument('--server', type=str, default="http://localhost:11434",
                       help='Ollama server URL (default: http://localhost:11434)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--retries', type=int, default=3,
                       help='Maximum number of retries (default: 3)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the images')
    
    args = parser.parse_args()
    
    # Run the test
    test_ollama_detector(
        image_url=args.url,
        model_name=args.model,
        server_url=args.server,
        timeout=args.timeout,
        max_retries=args.retries,
        save_image=not args.no_save
    )

if __name__ == "__main__":
    main()
