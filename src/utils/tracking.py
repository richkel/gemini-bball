"""
Tracking utilities for basketball shot analysis.

This module provides hand tracking and object tracking functionality
using MediaPipe and OpenCV.
"""

import cv2
import mediapipe as mp
import numpy as np
from enum import Enum, auto
from typing import Dict, Any, List, Tuple, Optional
import logging
import importlib.util

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure logging
logger = logging.getLogger(__name__)

class HandTracker:
    """Hand tracking using MediaPipe Hands."""
    
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        low_vram_mode: bool = True
    ):
        """Initialize the hand tracker.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            low_vram_mode: Enable optimizations for systems with <10GB VRAM
        """
        # In low VRAM mode, use more conservative settings
        if low_vram_mode:
            max_num_hands = min(max_num_hands, 1)  # Track at most 1 hand
            min_detection_confidence = max(min_detection_confidence, 0.6)  # Higher threshold
            min_tracking_confidence = max(min_tracking_confidence, 0.6)  # Higher threshold
        
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            # Use model complexity 0 (fastest) in low VRAM mode
            model_complexity=0 if low_vram_mode else 1
        )
        
        self.low_vram_mode = low_vram_mode
        
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a frame for hand tracking.
        
        Args:
            frame: Input frame as a numpy array (BGR format)
            
        Returns:
            Tuple of (annotated frame, hand landmarks)
        """
        # In low VRAM mode, downsample the frame first
        if hasattr(self, 'low_vram_mode') and self.low_vram_mode:
            # Resize to half resolution for processing
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, (w//2, h//2))
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            # Convert BGR to RGB at full resolution
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Draw hand landmarks with simplified style in low VRAM mode
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if hasattr(self, 'low_vram_mode') and self.low_vram_mode:
                    # Simplified drawing style for better performance
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                else:
                    # Full drawing style
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        # Return the annotated frame and hand landmarks
        return annotated_frame, {
            "multi_hand_landmarks": results.multi_hand_landmarks,
            "multi_handedness": results.multi_handedness,
            "hands_detected": results.multi_hand_landmarks is not None
        }
        
    def close(self):
        """Release resources safely."""
        try:
            if hasattr(self, 'hands') and self.hands is not None:
                self.hands.close()
        except ValueError as e:
            # Handle the case where the graph is already closed
            if "already None" in str(e):
                pass  # Graph already closed, nothing to do
            else:
                # Re-raise if it's a different ValueError
                raise


class BallType(Enum):
    """Enum for different ball types."""
    BASKETBALL = auto()
    TENNIS = auto()
    SOCCER = auto()
    VOLLEYBALL = auto()
    BASEBALL = auto()
    GENERIC = auto()
    
    @classmethod
    def from_string(cls, name: str) -> 'BallType':
        """Convert string to BallType enum."""
        name = name.upper().replace(" ", "")
        try:
            return cls[name]
        except KeyError:
            # Default to BASKETBALL if not found
            logger.warning(f"Unknown ball type: {name}, defaulting to BASKETBALL")
            return cls.BASKETBALL


class BallTracker:
    """Multi-sport ball tracking using OpenCV object tracking."""
    
    # Color thresholds for different ball types in HSV
    BALL_COLOR_THRESHOLDS = {
        BallType.BASKETBALL: [(0, 100, 100), (30, 255, 255)],     # Orange
        BallType.TENNIS: [(25, 100, 100), (40, 255, 255)],        # Yellow/green
        BallType.SOCCER: [(0, 0, 100), (180, 30, 255)],           # Primarily white with some black
        BallType.VOLLEYBALL: [(0, 0, 150), (180, 30, 255)],       # Primarily white
        BallType.BASEBALL: [(0, 0, 180), (180, 30, 255)],         # White
        BallType.GENERIC: [(0, 0, 50), (180, 255, 255)]           # Any color (very permissive)
    }
    
    def __init__(
        self,
        tracker_type: str = "CSRT",
        ball_type: BallType = BallType.BASKETBALL,
        min_ball_radius: int = 10,
        custom_color_lower: Optional[Tuple[int, int, int]] = None,
        custom_color_upper: Optional[Tuple[int, int, int]] = None
    ):
        """Initialize the ball tracker.
        
        Args:
            tracker_type: Type of OpenCV tracker to use
            ball_type: Type of ball to track
            min_ball_radius: Minimum radius for ball detection
            custom_color_lower: Optional custom lower bound for ball color in HSV
            custom_color_upper: Optional custom upper bound for ball color in HSV
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.tracking_initialized = False
        self.ball_bbox = None
        self.ball_type = ball_type
        self.min_ball_radius = min_ball_radius
        
        # Set color thresholds based on ball type or custom values
        if custom_color_lower is not None and custom_color_upper is not None:
            self.ball_color_lower = np.array(custom_color_lower)
            self.ball_color_upper = np.array(custom_color_upper)
        else:
            self.ball_color_lower = np.array(self.BALL_COLOR_THRESHOLDS[ball_type][0])
            self.ball_color_upper = np.array(self.BALL_COLOR_THRESHOLDS[ball_type][1])
            
        logger.info(f"Initialized ball tracker for {ball_type.name} ball")
        
    def _create_tracker(self):
        """Create a new tracker based on the specified type."""
        if self.tracker_type == "BOOSTING":
            return cv2.legacy.TrackerBoosting_create()
        elif self.tracker_type == "MIL":
            return cv2.legacy.TrackerMIL_create()
        elif self.tracker_type == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == "TLD":
            return cv2.legacy.TrackerTLD_create()
        elif self.tracker_type == "MEDIANFLOW":
            return cv2.legacy.TrackerMedianFlow_create()
        elif self.tracker_type == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        else:
            return cv2.legacy.TrackerCSRT_create()  # Default to CSRT
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a frame for ball tracking.
        
        Args:
            frame: Input frame as a numpy array (BGR format)
            
        Returns:
            Tuple of (annotated frame, tracking results)
        """
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Downsample frame for faster processing (optional based on frame size)
        h, w = frame.shape[:2]
        process_scale = 1.0
        process_frame = frame
        
        # If frame is large, downsample for processing
        if w > 640:
            process_scale = 640.0 / w
            process_frame = cv2.resize(frame, (640, int(h * process_scale)))
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)
        
        # Get the color range for the current ball type
        lower_color, upper_color = self._get_color_range()
        
        # Create a mask for the ball color
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour that could be a ball
        ball_position = None
        ball_radius = 0
        
        if contours:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Limit the number of contours to check for performance
            max_contours = 5
            for contour in contours[:max_contours]:
                # Approximate the contour to a circle
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                
                # Only consider contours with a minimum radius
                if radius > self.min_ball_radius:
                    # Calculate circularity
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Only consider circular contours
                    if circularity > 0.7:
                        # Scale back coordinates if we downsampled
                        if process_scale != 1.0:
                            x = x / process_scale
                            y = y / process_scale
                            radius = radius / process_scale
                            
                        ball_position = (int(x), int(y))
                        ball_radius = int(radius)
                        break
        
        tracking_info = {
            "ball_detected": False,
            "ball_bbox": None,
            "ball_center": None
        }
        
        # If ball is detected, update tracking info
        if ball_position is not None:
            tracking_info["ball_detected"] = True
            tracking_info["ball_center"] = ball_position
            
            # Draw the ball if found
            # Use thinner lines in low VRAM mode (determined by min_ball_radius setting)
            line_thickness = 1 if self.min_ball_radius > 10 else 2
            cv2.circle(annotated_frame, ball_position, ball_radius, (0, 255, 0), line_thickness)
            cv2.circle(annotated_frame, ball_position, 5, (0, 0, 255), -1)
            
            # Add text label (simplified in low VRAM mode)
            if self.min_ball_radius <= 10:  # Not in low VRAM mode
                label = f"{self.ball_type.name}: {ball_radius}px"
                cv2.putText(
                    annotated_frame,
                    label,
                    (ball_position[0] - 10, ball_position[1] - ball_radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        
        return annotated_frame, tracking_info
    
    def _get_color_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the HSV color range for the current ball type.
        
        Returns:
            Tuple of (lower_color, upper_color) HSV thresholds as numpy arrays
        """
        # Return the already initialized color thresholds as numpy arrays
        return self.ball_color_lower, self.ball_color_upper
        
    def reset(self):
        """Reset the tracker."""
        self.tracking_initialized = False
        self.tracker = None
        self.ball_bbox = None


class TrackingManager:
    """Manager for hand and ball tracking."""
    
    def __init__(
        self,
        enable_hand_tracking: bool = True,
        enable_ball_tracking: bool = True,
        max_num_hands: int = 2,
        ball_type: BallType = BallType.BASKETBALL,
        use_ollama_detection: bool = False,
        ollama_model: str = "gemma3:12b-it-q4_K_M",
        low_vram_mode: bool = True
    ):
        """Initialize the tracking manager.
        
        Args:
            enable_hand_tracking: Whether to enable hand tracking
            enable_ball_tracking: Whether to enable ball tracking
            max_num_hands: Maximum number of hands to detect
            ball_type: Type of ball to track
            use_ollama_detection: Whether to use Ollama for object detection
            ollama_model: Ollama model to use for object detection
            low_vram_mode: Enable optimizations for systems with <10GB VRAM
        """
        self.enable_hand_tracking = enable_hand_tracking
        self.enable_ball_tracking = enable_ball_tracking
        self.ball_type = ball_type
        self.use_ollama_detection = use_ollama_detection
        self.ollama_model = ollama_model
        self.low_vram_mode = low_vram_mode
        
        # Initialize trackers
        self.hand_tracker = None
        self.ball_tracker = None
        self.ollama_detector = None
        
        if enable_hand_tracking:
            self.hand_tracker = HandTracker(
                max_num_hands=max_num_hands,
                low_vram_mode=low_vram_mode,
                # Higher thresholds in low VRAM mode for better performance
                min_detection_confidence=0.6 if low_vram_mode else 0.5,
                min_tracking_confidence=0.6 if low_vram_mode else 0.5
            )
            
        if enable_ball_tracking:
            self.ball_tracker = BallTracker(
                ball_type=ball_type,
                # In low VRAM mode, use a larger minimum ball radius to reduce false positives
                min_ball_radius=15 if low_vram_mode else 10
            )
            
        # Initialize Ollama detector if enabled
        if use_ollama_detection:
            try:
                # Import OllamaObjectDetector dynamically to avoid circular imports
                spec = importlib.util.find_spec("src.utils.ollama_detector")
                if spec is not None:
                    ollama_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ollama_module)
                    self.ollama_detector = ollama_module.OllamaObjectDetector(model_name=ollama_model)
                    logger.info(f"Initialized Ollama object detector with model {ollama_model}")
                else:
                    logger.warning("Could not find Ollama detector module, disabling Ollama detection")
                    self.use_ollama_detection = False
            except Exception as e:
                logger.error(f"Failed to initialize Ollama detector: {e}")
                self.use_ollama_detection = False
            
    def set_ball_type(self, ball_type: BallType):
        """Change the ball type for tracking.
        
        Args:
            ball_type: New ball type to track
        """
        self.ball_type = ball_type
        
        # Reinitialize the ball tracker with the new ball type
        if self.enable_ball_tracking:
            self.ball_tracker = BallTracker(
                ball_type=ball_type,
                min_ball_radius=15 if self.low_vram_mode else 10
            )
            logger.info(f"Switched ball tracking to {ball_type.name}")
    
    # ... rest of the class remains the same ...
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a frame for tracking.
        
        Args:
            frame: Frame to process
            
        Returns:
            Annotated frame and tracking information
        """
        tracking_info = {
            "hands_detected": False,
            "hand_landmarks": [],
            "ball_detected": False,
            "ball_bbox": None,
            "ball_center": None,
            "ball_type": self.ball_type,
            "detection_method": "color"
        }
        
        annotated_frame = frame.copy()
        
        # Process hand tracking if enabled
        if self.enable_hand_tracking and self.hand_tracker is not None:
            annotated_frame, hand_info = self.hand_tracker.process(annotated_frame)
            tracking_info.update(hand_info)
        
        # Process ball tracking if enabled
        if self.enable_ball_tracking and self.ball_tracker is not None:
            annotated_frame, ball_info = self.ball_tracker.process(annotated_frame)
            tracking_info.update(ball_info)
            
            # If color-based tracking failed and Ollama detection is enabled, try that
            if not tracking_info["ball_detected"] and self.use_ollama_detection and self.ollama_detector is not None:
                try:
                    # Use Ollama for object detection
                    ollama_result = self.ollama_detector.detect_ball(frame)
                    
                    if ollama_result["detected"] and ollama_result["coordinates"]:
                        # Convert Ollama coordinates to color tracker format
                        x1, y1, x2, y2 = ollama_result["coordinates"]
                        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        
                        # Update tracking info
                        tracking_info["ball_detected"] = True
                        tracking_info["ball_bbox"] = bbox
                        tracking_info["ball_type"] = ollama_result.get("ball_type", self.ball_type)
                        tracking_info["ball_confidence"] = ollama_result.get("confidence", 0.0)
                        tracking_info["detection_method"] = "ollama"
                        
                        # Calculate ball center
                        center_x = x1 + (x2 - x1) // 2
                        center_y = y1 + (y2 - y1) // 2
                        tracking_info["ball_center"] = (int(center_x), int(center_y))
                        
                        # Process the frame with the ball tracker to get the visualization
                        self.ball_tracker.ball_bbox = bbox
                        annotated_frame, _ = self.ball_tracker.process_detection(annotated_frame, tracking_info)
                        
                        logger.info(f"Ollama detected {tracking_info['ball_type'].name if isinstance(tracking_info['ball_type'], BallType) else tracking_info['ball_type']} with confidence {tracking_info['ball_confidence']:.2f}")
                except Exception as e:
                    logger.error(f"Error using Ollama detector: {e}")
        
        return annotated_frame, tracking_info
    
    def close(self):
        """Release resources safely."""
        try:
            if self.hand_tracker is not None:
                self.hand_tracker.close()
        except Exception as e:
            # Log error but continue cleanup
            logger.error(f"Error closing hand tracker: {e}")
            
        try:
            if self.ball_tracker is not None:
                self.ball_tracker.reset()
        except Exception as e:
            # Log error but continue cleanup
            logger.error(f"Error resetting ball tracker: {e}")
