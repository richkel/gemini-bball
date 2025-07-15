"""
Test script for the Gemini backend.

This script tests the Gemini backend with a test frame.
"""

import os
import sys
import asyncio
import logging
import numpy as np
import cv2
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Gemini backend
from src.analysis.gemini_backend import GeminiBackend, ShotAnalysis, ShotResult, ShotType, ShotStatistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

async def test_gemini():
    """Test the Gemini backend with a test frame."""
    print("=== Gemini Basketball Shot Analyzer Test ===\n")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        return
    
    # Initialize the Gemini backend
    print(f"Initializing with Gemini...")
    gemini = GeminiBackend(api_key=api_key)
    
    try:
        # Create a test frame
        print("Creating test frame...")
        frame = create_test_frame()
        
        # Save the test frame for reference
        cv2.imwrite("test_frame.jpg", frame)
        print("Saved test frame as 'test_frame.jpg'")
        
        # Analyze the frame
        print("Running analysis...")
        analysis = await gemini.analyze_frame(frame, timestamp=0.0)
        
        if analysis:
            print("\n🎯 Analysis Results:")
            print(f"- Shot Type: {analysis.shot_type}")
            print(f"- Result: {analysis.result}")
            print(f"- Confidence: {analysis.confidence:.2f}")
            print(f"- Feedback: {analysis.feedback}")
            
            print("\n📊 Statistics:")
            print(f"- Total Shots: {gemini.stats.total_shots}")
            print(f"- Made: {gemini.stats.made_shots}")
            print(f"- Missed: {gemini.stats.missed_shots}")
            print(f"- FG%: {gemini.stats.field_goal_percentage:.1f}%")
        else:
            print("\nℹ️ No shot detected in the test frame.")
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        # Clean up
        await gemini.close()
    
    print("\n✅ Test completed.")

if __name__ == "__main__":
    asyncio.run(test_gemini())
