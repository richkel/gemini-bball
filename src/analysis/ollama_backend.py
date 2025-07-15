"""
Ollama VLM/Omni model integration for basketball shot analysis.
"""
import asyncio
import base64
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import httpx
from PIL import Image
from io import BytesIO
import sys
from pathlib import Path
import cv2

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from .base import AnalysisBackend
from ..models import ShotAnalysis, ShotResult, ShotType, ShotStatistics

logger = logging.getLogger(__name__)

class OllamaBackend(AnalysisBackend):
    """Basketball shot analysis backend using Ollama."""
    
    def __init__(
        self, 
        model_name: str = "llava:latest",
        base_url: str = "http://localhost:11434",
        low_vram_mode: bool = True,
        timeout: float = 30.0,
        **kwargs
    ):
        """Initialize the Ollama backend.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for the Ollama API
            low_vram_mode: Enable optimizations for systems with <10GB VRAM
            timeout: Timeout in seconds for API requests
            **kwargs: Additional arguments for the base class
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self.low_vram_mode = low_vram_mode
        self._ready = False
        
        logger.info(f"Initialized OllamaBackend with model {model_name}")
        if low_vram_mode:
            logger.info("Low VRAM mode enabled for Ollama - using optimized inference parameters")
        logger.info(f"Using timeout of {timeout} seconds for Ollama requests")
        
        # Default prompt template
        self.prompt_template = """
        You are an expert basketball coach analyzing a player's shooting form.
        
        Analyze the image and determine:
        1. If a shot was taken (look for shooting motion)
        2. If a shot was detected, analyze:
           - Shot result (made/missed/blocked)
           - Shot type (jump shot, three-pointer, layup, etc.)
           - Form feedback (what was good/bad about the shot)
        
        Previous shots: {shot_history}
        
        Respond with a JSON object containing:
        {{
            "shot_detected": boolean,
            "result": "made" or "missed" or "blocked" or null,
            "shot_type": string (e.g., "jump shot", "three-pointer", "layup"),
            "confidence": float (0.0 to 1.0),
            "feedback": string (detailed feedback on form),
            "reasoning": string (brief explanation of your analysis)
        }}
        """.strip()
    
    async def _ensure_model_available(self) -> bool:
        """Ensure the requested model is available."""
        if hasattr(self, '_ready') and self._ready:
            return True
            
        try:
            # Check if model is available
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            
            # Check if our model is in the list
            model_exists = any(m.get("name", "").startswith(self.model_name) for m in models)
            
            if not model_exists:
                logger.warning(f"Model {self.model_name} not found. Attempting to pull...")
                async with self.client.stream(
                    "POST", 
                    "/api/pull", 
                    json={"name": self.model_name}
                ) as response:
                    async for chunk in response.aiter_text():
                        # Stream the download progress
                        logger.debug(f"Downloading model: {chunk}")
            
            self._ready = True
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring model is available: {e}")
            self._ready = False
            return False
    
    def _format_shot_history(self) -> str:
        """Format shot history for the prompt."""
        if not self.shot_history:
            return "No previous shots in this session."
            
        recent_shots = self.shot_history[-5:]  # Last 5 shots
        return "\n".join(
            f"- {shot.shot_type}: {'✅' if shot.result == ShotResult.MADE else '❌'} "
            f"(Confidence: {shot.confidence:.2f})"
            for shot in recent_shots
        )
    
    async def analyze_frame(
        self, 
        frame: np.ndarray, 
        timestamp: float
    ) -> Optional[ShotAnalysis]:
        """Analyze a frame for shot detection and analysis using Ollama.
        
        Args:
            frame: Input frame as a numpy array (BGR format)
            timestamp: Timestamp of the frame in seconds
            
        Returns:
            ShotAnalysis object if a shot is detected, None otherwise
        """
        if not await self._ensure_model_available():
            logger.error("Ollama model is not available")
            return None
        
        try:
            # Encode the image as base64 with reduced quality in low VRAM mode
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85 if self.low_vram_mode else 95]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create the prompt
            prompt = self.prompt_template.format(
                shot_history=self._format_shot_history()
            )
            
            # Prepare inference options based on VRAM availability
            options = {}
            if self.low_vram_mode:
                options = {
                    "temperature": 0.1,  # Lower temperature for faster inference
                    "top_p": 0.7,      # Lower top_p for faster sampling
                    "num_ctx": 1024,   # Smaller context window
                    "num_gpu": 1,      # Ensure GPU acceleration
                    "num_thread": 4    # Optimize thread usage
                }
            
            try:
                # Send the request to Ollama with our configured timeout
                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "images": [image_base64],
                        "options": options
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Parse the response
                analysis = json.loads(result.get("response", "{}"))
                
                # Handle cases where Ollama didn't detect anything
                if not analysis.get("shot_detected", False):
                    return None
                
                # Map the response to our ShotAnalysis model
                shot_analysis = ShotAnalysis(
                    timestamp=timestamp,
                    result=ShotResult(analysis.get("result", "missed").lower()),
                    shot_type=ShotType.from_string(analysis.get("shot_type", "jump shot")),
                    confidence=float(analysis.get("confidence", 0.5)),
                    feedback=analysis.get("feedback", "No feedback provided"),
                    metadata={
                        "reasoning": analysis.get("reasoning", ""),
                        "raw_response": result,
                        "original_shot_type": analysis.get("shot_type", "")
                    }
                )
                return shot_analysis
                
            except httpx.TimeoutException:
                logger.warning(f"Ollama request timed out after {self.timeout} seconds")
                return None
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse Ollama response: {e}")
                logger.debug(f"Raw response: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing frame with Ollama: {e}", exc_info=True)
            return None
    
    @property
    def is_ready(self) -> bool:
        """Check if the backend is ready to process frames."""
        return hasattr(self, '_ready') and self._ready
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.client.aclose()
        self._ready = False

# Example usage:
# backend = OllamaBackend(model_name="llava:latest")
# shot = await backend.analyze_frame(frame, timestamp=0.0)
