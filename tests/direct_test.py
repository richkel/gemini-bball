"""
Direct test of Ollama integration for basketball shot analysis.
This is a self-contained script that doesn't rely on the package structure.
"""
import asyncio
import base64
import json
import logging
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import httpx
from PIL import Image
from io import BytesIO
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data Models
class ShotResult(str, Enum):
    MADE = "made"
    MISSED = "missed"
    BLOCKED = "blocked"
    FOUL = "foul"

class ShotType(str, Enum):
    JUMP_SHOT = "Jump shot"
    THREE_POINTER = "Three-pointer"
    LAYUP = "Layup"
    DUNK = "Dunk"
    FREE_THROW = "Free throw"
    HOOK_SHOT = "Hook shot"
    FLOATER = "Floater"
    PULL_UP = "Pull-up jump shot"
    STEP_BACK = "Step-back jump shot"

@dataclass
class ShotAnalysis:
    timestamp: float
    result: ShotResult
    shot_type: ShotType
    confidence: float
    feedback: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ShotStatistics:
    total_shots: int = 0
    made_shots: int = 0
    missed_shots: int = 0
    field_goal_percentage: float = 0.0
    shot_distribution: Dict[ShotType, int] = field(default_factory=dict)
    last_shot_time: Optional[float] = None

    def update(self, shot: ShotAnalysis):
        self.total_shots += 1
        
        if shot.result == ShotResult.MADE:
            self.made_shots += 1
        else:
            self.missed_shots += 1
            
        self.field_goal_percentage = (self.made_shots / self.total_shots) * 100 if self.total_shots > 0 else 0
        self.shot_distribution[shot.shot_type] = self.shot_distribution.get(shot.shot_type, 0) + 1
        self.last_shot_time = shot.timestamp

# Ollama Backend
class OllamaBackend:
    def __init__(self, model_name: str = "llava:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        # Set a short timeout for testing
        self.client = httpx.AsyncClient(
            base_url=base_url, 
            timeout=httpx.Timeout(10.0, read=30.0),
            http2=True
        )
        logger.info(f"Initialized OllamaBackend with model: {model_name}")
        self._ready = False
        self.stats = ShotStatistics()
        self.shot_history: List[ShotAnalysis] = []
        
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
        if self._ready:
            return True
            
        try:
            logger.info(f"Checking if model {self.model_name} is available...")
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            
            logger.info(f"Available models: {[m.get('name', '') for m in models]}")
            model_exists = any(m.get("name", "").startswith(self.model_name.split(':')[0]) for m in models)
            logger.info(f"Model {self.model_name} exists: {model_exists}")
            
            if not model_exists:
                logger.warning(f"Model {self.model_name} not found. Attempting to pull...")
                async with self.client.stream(
                    "POST", 
                    "/api/pull", 
                    json={"name": self.model_name}
                ) as response:
                    async for chunk in response.aiter_text():
                        logger.debug(f"Downloading model: {chunk}")
            
            self._ready = True
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring model is available: {e}")
            self._ready = False
            return False
    
    def _format_shot_history(self) -> str:
        if not self.shot_history:
            return "No previous shots in this session."
            
        recent_shots = self.shot_history[-5:]
        return "\n".join(
            f"- {shot.shot_type}: {'✅' if shot.result == ShotResult.MADE else '❌'} "
            f"(Confidence: {shot.confidence:.2f})"
            for shot in recent_shots
        )
    
    async def analyze_frame(self, frame: np.ndarray, timestamp: float) -> Optional[ShotAnalysis]:
        logger.info("Starting frame analysis...")
        try:
            if not await self._ensure_model_available():
                logger.error("Ollama model is not available")
                return None
        except Exception as e:
            logger.error(f"Error checking model availability: {e}", exc_info=True)
            return None
        
        try:
            # Convert frame to base64
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Prepare a simpler prompt for testing
            prompt = """
            Analyze this basketball shot image and return a JSON object with:
            {
                "shot_detected": true/false,
                "result": "made" or "missed" or "blocked" or null,
                "shot_type": string (e.g., "jump shot", "three-pointer", "layup"),
                "confidence": float (0.0 to 1.0),
                "feedback": string,
                "reasoning": string
            }
            """.strip()
            
            # Make the API call to Ollama with RTX 3080 optimized parameters
            logger.info("Sending request to Ollama API (RTX 3080 optimized)...")
            try:
                response = await asyncio.wait_for(
                    self.client.post(
                        "/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "images": [img_base64],
                            "stream": False,
                            "format": "json",
                            "options": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "num_ctx": 2048,  # Increased context for better understanding
                                "num_gpu": 1,     # Ensure GPU is being used
                                "num_thread": 8,   # Optimize for RTX 3080
                                "num_batch": 4     # Batch size for better throughput
                            }
                        },
                        timeout=300.0  # Increased timeout to 300 seconds for first run
                    ),
                    timeout=310.0  # Slightly higher than the request timeout
                )
                
                response.raise_for_status()
                result = response.json()
                logger.debug(f"Raw response: {result}")
                
                if not result or "response" not in result:
                    logger.error("Invalid response format from Ollama API")
                    return None
                    
                return await self._process_ollama_response(result)
                    
            except asyncio.TimeoutError:
                logger.error("Request to Ollama API timed out")
                return None
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error from Ollama API: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error calling Ollama API: {e}", exc_info=True)
                return None
            
            # This code is now handled in _process_ollama_response
            pass
                
        except Exception as e:
            logger.error(f"Error analyzing frame with Ollama: {e}", exc_info=True)
            return None
    
    @property
    def is_ready(self) -> bool:
        return self._ready
    
    async def _process_ollama_response(self, result: Dict[str, Any]) -> Optional[ShotAnalysis]:
        """Process the response from the Ollama API."""
        try:
            # Get the response text
            response_text = result.get("response", "{}")
            if not response_text.strip():
                logger.warning("Empty response from model")
                return None
                
            # Sometimes the response might be wrapped in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            logger.debug(f"Parsing response: {response_text}")
            
            # Parse the JSON response
            analysis = json.loads(response_text)
            
            # Check if a shot was detected
            if not analysis.get("shot_detected", False):
                return None
            
            # Safely get and validate the result
            result_value = analysis.get("result")
            if result_value is None:
                logger.warning("No result in analysis, defaulting to 'missed'")
                result_value = "missed"
            
            # Map model's shot type to our ShotType enum
            shot_type_map = {
                'jump shot': 'JUMP_SHOT',
                'jumper': 'JUMP_SHOT',
                'layup': 'LAYUP',
                'dunk': 'DUNK',
                'three pointer': 'THREE_POINTER',
                '3 pointer': 'THREE_POINTER',
                'free throw': 'FREE_THROW',
                'hook shot': 'HOOK_SHOT',
                'floater': 'FLOATER',
                'alley oop': 'ALLEY_OOP',
                'fadeaway': 'FADEAWAY',
                'bank shot': 'BANK_SHOT',
                'tip in': 'TIP_IN',
                'putback': 'PUTBACK'
            }
            
            # Get and clean the shot type from the model
            shot_type_value = analysis.get("shot_type", "").lower().strip()
            if not shot_type_value:
                logger.warning("No shot type in analysis, defaulting to 'JUMP_SHOT'")
                shot_type_value = "jump shot"
                
            # Map to our enum values
            shot_type_enum = shot_type_map.get(shot_type_value, "JUMP_SHOT")
            logger.info(f"Mapped shot type: '{shot_type_value}' -> {shot_type_enum}")
            
            # Safely get and validate confidence
            try:
                confidence = float(analysis.get("confidence", 0.5))
            except (TypeError, ValueError):
                logger.warning("Invalid confidence value, defaulting to 0.5")
                confidence = 0.5
            
            # Map the response to our ShotAnalysis model
            return ShotAnalysis(
                timestamp=time.time(),
                result=ShotResult(str(result_value).lower()),
                shot_type=ShotType[shot_type_enum],
                confidence=confidence,
                feedback=analysis.get("feedback", "No feedback provided"),
                metadata={
                    "reasoning": analysis.get("reasoning", ""),
                    "raw_response": result
                }
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Error processing Ollama response: {e}", exc_info=True)
            return None
    
    async def close(self) -> None:
        await self.client.aclose()
        self._ready = False

# Test Function
def create_test_frame(width=640, height=480):
    """Create a test frame that simulates a basketball shot."""
    # Create a blue background (like a basketball court)
    frame = np.full((height, width, 3), [50, 100, 200], dtype=np.uint8)
    
    # Draw a basketball hoop (red rectangle)
    hoop_height = 30
    hoop_width = 150
    hoop_x = width - 100
    hoop_y = 100
    cv2.rectangle(frame, 
                 (hoop_x, hoop_y), 
                 (hoop_x + hoop_width, hoop_y + hoop_height), 
                 (0, 0, 255), 2)
    
    # Draw a backboard (white rectangle)
    backboard_height = 100
    backboard_width = 200
    backboard_x = hoop_x - 25
    backboard_y = hoop_y - 20
    cv2.rectangle(frame, 
                 (backboard_x, backboard_y), 
                 (backboard_x + backboard_width, backboard_y + backboard_height), 
                 (255, 255, 255), -1)
    
    # Draw a basketball (orange circle)
    ball_radius = 20
    ball_x = width // 3
    ball_y = height // 2
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 165, 255), -1)
    
    # Draw a player (simple stick figure)
    player_x = ball_x - 50
    player_y = height - 50
    cv2.circle(frame, (player_x, player_y - 20), 10, (255, 255, 255), -1)  # Head
    cv2.line(frame, (player_x, player_y - 10), (player_x, player_y + 20), (255, 255, 255), 2)  # Body
    cv2.line(frame, (player_x, player_y), (player_x - 20, player_y + 10), (255, 255, 255), 2)  # Arm
    cv2.line(frame, (player_x, player_y + 20), (player_x - 20, player_y + 50), (255, 255, 255), 2)  # Leg 1
    cv2.line(frame, (player_x, player_y + 20), (player_x + 20, player_y + 50), (255, 255, 255), 2)  # Leg 2
    
    # Draw a shooting line (dashed)
    for i in range(0, 10):
        start_x = ball_x + int((hoop_x - ball_x) * i / 10)
        start_y = ball_y + int((hoop_y - ball_y) * i / 10)
        end_x = ball_x + int((hoop_x - ball_x) * (i + 0.5) / 10)
        end_y = ball_y + int((hoop_y - ball_y) * (i + 0.5) / 10)
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    return frame

async def main():
    print("=== Basketball Shot Analyzer Direct Test ===\n")
    
    # Use llava model which is already installed
    print("Initializing with llava:latest...")
    backend = OllamaBackend(model_name="llava:latest")
    
    print("Creating test frame...")
    test_frame = create_test_frame()
    
    print("Running analysis...")
    try:
        result = await backend.analyze_frame(test_frame, timestamp=0.0)
        if result:
            print("\n🎯 Analysis Results:")
            print(f"- Shot Type: {result.shot_type}")
            print(f"- Result: {result.result}")
            print(f"- Confidence: {result.confidence:.2f}")
            print(f"- Feedback: {result.feedback}")
            
            print("\n📊 Statistics:")
            print(f"- Total Shots: {backend.stats.total_shots}")
            print(f"- Made: {backend.stats.made_shots}")
            print(f"- Missed: {backend.stats.missed_shots}")
            print(f"- FG%: {backend.stats.field_goal_percentage:.1f}%")
        else:
            print("\nℹ️ No shot detected in the test frame.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await backend.close()
        print("\n✅ Test completed.")

if __name__ == "__main__":
    asyncio.run(main())
