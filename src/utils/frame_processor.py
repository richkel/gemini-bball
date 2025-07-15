"""
Utility functions for processing video frames.
"""
from typing import Tuple, Optional
import cv2
import numpy as np

def preprocess_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> np.ndarray:
    """Preprocess a frame for analysis.
    
    Args:
        frame: Input frame as a numpy array (BGR format)
        target_size: Optional target size as (width, height)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed frame
    """
    # Convert to RGB (most models expect RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize if target size is provided
    if target_size is not None:
        frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize if requested
    if normalize:
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
    
    return frame_rgb

def draw_shot_result(
    frame: np.ndarray,
    shot_analysis: 'ShotAnalysis',
    stats: 'ShotStatistics',
    position: Tuple[int, int] = (20, 40),
    font_scale: float = 0.7,
    thickness: int = 2,
    padding: int = 10,
    alpha: float = 0.7
) -> np.ndarray:
    """Draw shot analysis results on a frame.
    
    Args:
        frame: Input frame to draw on
        shot_analysis: Analysis results to display
        stats: Current shot statistics
        position: Top-left position of the text overlay
        font_scale: Font scale factor
        thickness: Text thickness
        padding: Padding around text
        alpha: Transparency of the background (0.0 to 1.0)
        
    Returns:
        Frame with analysis overlay
    """
    # Create a copy of the frame to draw on
    overlay = frame.copy()
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(40 * font_scale)
    
    # Define colors
    color = (0, 255, 0) if shot_analysis.result == "made" else (0, 0, 255)  # Green for made, red for missed
    bg_color = (0, 0, 0)  # Black background
    
    # Prepare text lines
    lines = [
        f"Shot: {shot_analysis.shot_type.value}",
        f"Result: {shot_analysis.result.value.upper()}",
        f"Confidence: {shot_analysis.confidence:.2f}",
        "",
        f"Made: {stats.made_shots} / {stats.total_shots}",
        f"FG%: {stats.field_goal_percentage:.1f}%",
        "",
        "Feedback:",
        *textwrap.wrap(shot_analysis.feedback, width=40)
    ]
    
    # Calculate text block size
    max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines if line)
    total_height = line_height * len([line for line in lines if line])
    
    # Draw semi-transparent background
    cv2.rectangle(
        overlay,
        (x - padding, y - padding),
        (x + max_width + padding, y + total_height + padding),
        bg_color,
        -1
    )
    
    # Add the text
    for i, line in enumerate(lines):
        if not line:
            y += line_height
            continue
            
        # Use different color for the result line
        line_color = color if i == 1 else (255, 255, 255)  # White for all other text
        
        cv2.putText(
            overlay,
            line,
            (x, y + i * line_height),
            font,
            font_scale,
            line_color,
            thickness,
            cv2.LINE_AA
        )
    
    # Blend the overlay with the original frame
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
