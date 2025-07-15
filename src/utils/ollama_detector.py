"""
Object detection using Ollama models.

This module provides a class for detecting objects in images using Ollama models.
"""
import logging
import requests
import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import httpx
import numpy as np
import requests

from ..models.shot_models import BallType

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class OllamaObjectDetector:
    """Object detection using Ollama."""
    
    def __init__(self, model_name: str = "gemma3:12b-it-q4_K_M", low_vram_mode: bool = True):
        """Initialize the Ollama object detector.
        
        Args:
            model_name: Name of the Ollama model to use
            low_vram_mode: Enable optimizations for systems with <10GB VRAM
        """
        self.model_name = model_name
        self.client = httpx.AsyncClient()
        self.base_url = "http://localhost:11434"
        self.low_vram_mode = low_vram_mode
        self.max_retries = 3
        self.timeout = 30.0
        
        if low_vram_mode:
            logger.info(f"OllamaObjectDetector initialized with {model_name} in low VRAM mode")
    
    async def _encode_image(self, frame: np.ndarray) -> str:
        """Encode an image as a base64 string.
        
        Args:
            frame: Image as a numpy array (BGR format)
            
        Returns:
            Base64 encoded image string
        """
        # Resize large images to reduce payload size
        h, w = frame.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            frame = cv2.resize(frame, new_size)
        
        # Encode the image
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80 if self.low_vram_mode else 95]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return base64.b64encode(buffer).decode('utf-8')
    
    async def detect_ball(self, frame: np.ndarray, ball_type: BallType) -> bool:
        """Detect if a specific ball type is present in the frame.
        
        Args:
            frame: Input frame as a numpy array (BGR format)
            ball_type: Type of ball to detect
            
        Returns:
            True if the ball is detected, False otherwise
        """
        # Prepare inference options based on VRAM availability
        options = {}
        if self.low_vram_mode:
            options = {
                "temperature": 0.1,  # Lower temperature for faster inference
                "top_p": 0.7,      # Lower top_p for faster sampling
                "num_ctx": 512,    # Much smaller context window for detection
                "num_gpu": 1,      # Ensure GPU acceleration
                "num_thread": 4    # Optimize thread usage
            }
        
        # Convert frame to base64 with reduced quality in low VRAM mode
        image_base64 = await self._encode_image(frame)
        
        # Create a simpler prompt in low VRAM mode
        if self.low_vram_mode:
            prompt = f"Is there a {ball_type.name.lower()} in this image? Answer YES or NO only."
        else:
            prompt = f"""Look at this image and tell me if there is a {ball_type.name.lower()} in it. 
            Only respond with YES or NO followed by a brief explanation."""
        
        try:
            # Send the request to Ollama
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "images": [image_base64],
                    "options": options
                },
                timeout=15.0 if self.low_vram_mode else 30.0  # Shorter timeout in low VRAM mode
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the response text
            text = result.get("response", "")
            
            # Check if the response indicates a ball was detected
            if re.search(r"\byes\b", text.lower()):
                logger.info(f"Detected {ball_type.name} in frame")
                return True
            else:
                logger.debug(f"No {ball_type.name} detected in frame")
                return False
                
        except Exception as e:
            logger.error(f"Error detecting ball: {e}")
            return False
            
    
    def integrate_with_color_tracker(
        self, 
        frame: np.ndarray, 
        color_tracker_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate Ollama detection with color-based tracking.
        
        This method combines the results of color-based tracking with Ollama detection.
        If color tracking detected a ball, we use those results. Otherwise, we try
        Ollama detection as a fallback.
        
        Args:
            frame: Image frame
            color_tracker_result: Results from color-based tracking
            
        Returns:
            Combined tracking results
        """
        # If color tracking detected a ball, use those results
        if color_tracker_result.get("ball_detected", False):
            return color_tracker_result
        
        # Otherwise, try Ollama detection
        ollama_result = self.detect_ball(frame)
        
        if ollama_result["detected"] and ollama_result["coordinates"]:
            # Convert Ollama coordinates to color tracker format
            x1, y1, x2, y2 = ollama_result["coordinates"]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            
            # Update color tracker result
            color_tracker_result["ball_detected"] = True
            color_tracker_result["ball_bbox"] = bbox
            color_tracker_result["ball_type"] = ollama_result["ball_type"]
            color_tracker_result["ball_confidence"] = ollama_result["confidence"]
            
            # Calculate ball center
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            color_tracker_result["ball_center"] = (int(center_x), int(center_y))
        
        return color_tracker_result
