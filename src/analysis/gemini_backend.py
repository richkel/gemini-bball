"""
Gemini Backend for Basketball Shot Analysis

This module provides an implementation of the AnalysisBackend interface
that uses Google's Gemini API for analyzing basketball shots.
"""

import base64
import json
import logging
from typing import Dict, Any, Optional, List
import httpx
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

from .base import AnalysisBackend
from ..models import ShotAnalysis, ShotResult, ShotType, ShotStatistics

logger = logging.getLogger(__name__)

class GeminiBackend(AnalysisBackend):
    """Implementation of AnalysisBackend using Google's Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro-vision"):
        """Initialize the Gemini backend.
        
        Args:
            api_key: Google Cloud API key with access to the Gemini API
            model_name: The name of the Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            params={"key": self.api_key},
            timeout=60.0
        )
        self.shot_history: List[ShotAnalysis] = []
        self.stats = ShotStatistics()
        
        logger.info(f"Initialized GeminiBackend with model: {self.model_name}")
    
    async def _encode_image(self, frame: np.ndarray) -> str:
        """Encode a frame as a base64 string."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_img = Image.fromarray(rgb_frame)
        # Convert to bytes
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        # Encode as base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    async def _process_gemini_response(self, response: Dict[str, Any]) -> Optional[ShotAnalysis]:
        """Process the response from the Gemini API."""
        try:
            # Extract the text content from the response
            text_content = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            
            # Parse the JSON response
            try:
                analysis = json.loads(text_content.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in text_content:
                    text_content = text_content.split("```json")[1].split("```")[0].strip()
                elif "```" in text_content:
                    text_content = text_content.split("```")[1].split("```")[0].strip()
                analysis = json.loads(text_content)
            
            # Check if a shot was detected
            if not analysis.get("shot_detected", False):
                return None
            
            # Map the response to our ShotAnalysis model
            return ShotAnalysis(
                timestamp=time.time(),
                result=ShotResult(analysis.get("result", "missed").lower()),
                shot_type=ShotType(analysis.get("shot_type", "jump_shot").upper()),
                confidence=float(analysis.get("confidence", 0.5)),
                feedback=analysis.get("feedback", "No feedback provided"),
                metadata={
                    "reasoning": analysis.get("reasoning", ""),
                    "raw_response": response
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing Gemini response: {e}", exc_info=True)
            return None
    
    async def analyze_frame(self, frame: np.ndarray, timestamp: float) -> Optional[ShotAnalysis]:
        """Analyze a frame to detect and analyze basketball shots.
        
        Args:
            frame: The frame to analyze (numpy array in BGR format)
            timestamp: The timestamp of the frame in the video
            
        Returns:
            ShotAnalysis object if a shot is detected, None otherwise
        """
        try:
            # Encode the frame as base64
            img_base64 = await self._encode_image(frame)
            
            # Prepare the prompt
            prompt = """
            Analyze this basketball frame and provide a JSON response with the following structure:
            {
                "shot_detected": boolean,
                "shot_type": string (e.g., "jump_shot", "layup", "three_pointer"),
                "result": string ("made" or "missed"),
                "confidence": float (0-1),
                "feedback": string (brief feedback on the shot),
                "reasoning": string (brief explanation of the analysis)
            }
            
            Guidelines:
            - Only include the JSON in your response
            - shot_type should be one of: jump_shot, layup, three_pointer, free_throw, hook_shot, floater, alley_oop, fadeaway, bank_shot, tip_in, putback
            - result should be "made" or "missed"
            - confidence should be between 0 and 1
            - feedback should be a brief sentence
            - reasoning should explain your analysis
            """
            
            # Prepare the request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_base64
                                }
                            }
                        ]
                    }
                ],
                "generation_config": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
            }
            
            # Make the API call
            logger.info("Sending request to Gemini API...")
            response = await self.client.post(
                f"/models/{self.model_name}:generateContent",
                json=payload
            )
            response.raise_for_status()
            
            # Process the response
            result = await self._process_gemini_response(response.json())
            
            if result:
                # Update statistics
                self.shot_history.append(result)
                self.stats.update(result)
                
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing frame with Gemini: {e}", exc_info=True)
            return None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.info("Gemini backend closed")
