#!/usr/bin/env python
"""
Basketball Shot Analyzer with Real-time UI Controls

This script runs the webcam analyzer with a UI for adjusting settings in real-time.
"""

import argparse
import logging
import cv2
import time
from typing import Dict, Any, Optional

# Import relative modules
from run_webcam_analyzer import WebcamAnalyzer
from src.pipeline.real_time_analyzer import BackendType
from src.utils.tracking import BallType
from src.utils.ui_controls import SettingsUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class WebcamAnalyzerWithUI:
    """Webcam analyzer with real-time UI controls."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the webcam analyzer with UI.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        self.analyzer: Optional[WebcamAnalyzer] = None
        self.ui = SettingsUI("Basketball Analyzer Settings")
        self.setup_ui()
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        
    def setup_ui(self):
        """Set up the UI controls."""
        # Add performance controls
        self.ui.add_toggle("low_vram_mode", self.args.low_vram, self.toggle_low_vram)
        self.ui.add_slider("frame_skip", 1, 30, self.args.frame_skip, self.set_frame_skip)
        self.ui.add_slider("resolution_scale", 0.25, 1.0, self.args.resolution_scale, self.set_resolution_scale)
        
        # Add Ollama timeout settings
        self.ui.add_slider("ollama_timeout", 10.0, 60.0, self.args.ollama_timeout, self.set_ollama_timeout)
        self.ui.add_slider("ollama_retry_count", 1, 5, self.args.ollama_retry_count, self.set_ollama_retry_count)
        self.ui.add_slider("ollama_retry_delay", 0.5, 5.0, self.args.ollama_retry_delay, self.set_ollama_retry_delay)
        
        # Add tracking controls
        self.ui.add_toggle("hand_tracking", not self.args.no_hand_tracking, self.toggle_hand_tracking)
        self.ui.add_toggle("ball_tracking", True, self.toggle_ball_tracking)
        
        # Ball type dropdown
        ball_types = ["BASKETBALL", "TENNIS", "SOCCER", "VOLLEYBALL", "BASEBALL", "GENERIC"]
        current_ball_idx = ball_types.index(self.args.ball_type)
        self.ui.add_slider("ball_type_idx", 0, len(ball_types) - 1, current_ball_idx, self.set_ball_type)
        
        # Add backend selection
        self.ui.add_toggle("use_ollama", self.args.backend == "ollama", self.toggle_backend)
        
        # Add action buttons
        self.ui.add_button("save_frame", self.save_frame)
        self.ui.add_button("reset_defaults", self.reset_defaults)
    
    def toggle_low_vram(self, enabled: bool):
        """Toggle low VRAM mode.
        
        Args:
            enabled: Whether to enable low VRAM mode
        """
        if self.analyzer:
            logger.info(f"Setting low VRAM mode: {enabled}")
            self.analyzer.low_vram = enabled
    
    def set_frame_skip(self, value: float):
        """Set frame skip value.
        
        Args:
            value: Frame skip value
        """
        if self.analyzer:
            frame_skip = int(value)
            logger.info(f"Setting frame skip: {frame_skip}")
            self.analyzer.frame_skip = frame_skip
    
    def set_resolution_scale(self, value: float):
        """Set resolution scale.
        
        Args:
            value: Resolution scale (0.25-1.0)
        """
        if self.analyzer:
            logger.info(f"Setting resolution scale: {value:.2f}")
            self.analyzer.resolution_scale = value
    
    def toggle_hand_tracking(self, enabled: bool):
        """Toggle hand tracking.
        
        Args:
            enabled: Whether to enable hand tracking
        """
        if self.analyzer:
            logger.info(f"Setting hand tracking: {enabled}")
            self.analyzer.enable_hand_tracking = enabled
    
    def toggle_ball_tracking(self, enabled: bool):
        """Toggle ball tracking.
        
        Args:
            enabled: Whether to enable ball tracking
        """
        if self.analyzer:
            logger.info(f"Setting ball tracking: {enabled}")
            self.analyzer.enable_ball_tracking = enabled
    
    def set_ball_type(self, value: int) -> None:
        """Set the ball type.
        
        Args:
            value: Ball type index
        """
        ball_types = ["BASKETBALL", "TENNIS", "SOCCER", "VOLLEYBALL", "BASEBALL", "GENERIC"]
        self.args.ball_type = ball_types[value]
        logger.info(f"Ball type set to {self.args.ball_type}")
        
        # Update the analyzer's ball type if it exists
        if hasattr(self, 'analyzer') and self.analyzer:
            self.analyzer.ball_type = BallType.from_string(self.args.ball_type)
    
    def toggle_backend(self, use_ollama: bool):
        """Toggle backend between Ollama and color-based.
        
        Args:
            use_ollama: Whether to use Ollama backend
        """
        if hasattr(self, 'analyzer') and self.analyzer:
            backend = "ollama" if use_ollama else "gemini"
            logger.info(f"Setting backend: {backend}")
            self.analyzer.backend_type = backend
    
    def set_ollama_timeout(self, value: float):
        """Set Ollama request timeout.
        
        Args:
            value: Timeout value in seconds
        """
        if self.analyzer:
            logger.info(f"Setting Ollama timeout: {value:.1f} seconds")
            self.analyzer.ollama_timeout = value
            
    def set_ollama_retry_count(self, value: float):
        """Set Ollama retry count.
        
        Args:
            value: Number of retries
        """
        if self.analyzer:
            retry_count = int(value)
            logger.info(f"Setting Ollama retry count: {retry_count}")
            self.analyzer.ollama_retry_count = retry_count
            
    def set_ollama_retry_delay(self, value: float):
        """Set Ollama retry delay.
        
        Args:
            value: Delay between retries in seconds
        """
        if self.analyzer:
            logger.info(f"Setting Ollama retry delay: {value:.1f} seconds")
            self.analyzer.ollama_retry_delay = value
    
    def save_frame(self):
        """Save the current frame."""
        if self.analyzer and hasattr(self.analyzer, 'current_frame') and self.analyzer.current_frame is not None:
            filename = f"frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.analyzer.current_frame)
            logger.info(f"Saved frame to {filename}")
    
    def reset_defaults(self):
        """Reset settings to defaults."""
        if self.analyzer:
            logger.info("Resetting to default settings")
            self.ui.set_value("low_vram_mode", self.args.low_vram)
            self.ui.set_value("frame_skip", self.args.frame_skip)
            self.ui.set_value("resolution_scale", self.args.resolution_scale)
            self.ui.set_value("hand_tracking", not self.args.no_hand_tracking)
            self.ui.set_value("ball_tracking", True)
            
            ball_types = [bt.name for bt in BallType]
            current_ball_idx = ball_types.index(self.args.ball_type.name)
            self.ui.set_value("ball_type_idx", current_ball_idx)
            
            self.ui.set_value("use_ollama", self.args.backend == BackendType.OLLAMA.name)
    
    async def run(self):
        """Run the webcam analyzer with UI."""
        # Initialize the webcam analyzer
        self.analyzer = WebcamAnalyzer(
            camera_index=self.args.camera,
            backend_type=self.args.backend,
            model_name=self.args.model,
            ball_type=self.args.ball_type,
            frame_skip=self.args.frame_skip,
            enable_hand_tracking=not self.args.no_hand_tracking,
            low_vram_mode=self.args.low_vram,
            resolution_scale=self.args.resolution_scale,
            ollama_timeout=self.args.ollama_timeout,
            ollama_retry_count=self.args.ollama_retry_count,
            ollama_retry_delay=self.args.ollama_retry_delay
        )
        
        # Initialize the analyzer
        await self.analyzer.initialize()
        
        # Show the UI
        self.ui.show()
        
        try:
            # Process frames until stopped
            while True:
                # Get a frame from the webcam
                ret, frame = self.analyzer.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Store the current frame for saving
                self.analyzer.current_frame = frame.copy()
                
                # Create a copy of the frame for UI overlays
                display_frame = frame.copy()
                
                # Calculate FPS
                current_time = time.time()
                self.frame_count += 1
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time
                
                # Add UI overlays to display frame (not processed frame)
                cv2.putText(
                    display_frame, 
                    f"FPS: {self.fps}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                cv2.putText(
                    display_frame, 
                    "Press 'u' to toggle settings UI", 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Process the frame without UI overlays
                processed_frame = await self.analyzer.process_frame(frame)
                
                # Ensure both frames have the same size
                if processed_frame.shape != display_frame.shape:
                    # Resize display_frame to match processed_frame
                    display_frame = cv2.resize(display_frame, 
                                             (processed_frame.shape[1], processed_frame.shape[0]))
                
                # Combine the processed frame with UI overlays
                overlay = cv2.addWeighted(processed_frame, 1, display_frame, 0.7, 0)
                
                # Display the combined frame
                cv2.imshow("Basketball Shot Analyzer", overlay)
                
                # Update the UI
                self.ui.update()
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                # 'q' to quit
                if key == ord('q'):
                    break
                
                # 'u' to toggle UI
                elif key == ord('u'):
                    self.ui.toggle_visibility()
                
                # 's' to save frame
                elif key == ord('s'):
                    self.save_frame()
                
                # 'b' to cycle ball types
                elif key == ord('b'):
                    ball_types = list(BallType)
                    current_idx = ball_types.index(self.analyzer.ball_type)
                    next_idx = (current_idx + 1) % len(ball_types)
                    self.analyzer.ball_type = ball_types[next_idx]
                    self.ui.set_value("ball_type_idx", next_idx)
                    logger.info(f"Switched to ball type: {ball_types[next_idx].name}")
                
                # 'o' to toggle Ollama detection
                elif key == ord('o'):
                    use_ollama = self.analyzer.backend_type == BackendType.OLLAMA
                    new_backend = BackendType.COLOR if use_ollama else BackendType.OLLAMA
                    self.analyzer.backend_type = new_backend
                    self.ui.set_value("use_ollama", new_backend == BackendType.OLLAMA)
                    logger.info(f"Switched to backend: {new_backend.name}")
                
                # 'r' to reset defaults
                elif key == ord('r'):
                    self.reset_defaults()
        
        finally:
            # Clean up
            if self.analyzer:
                if hasattr(self.analyzer, 'cleanup'):
                    cleanup_method = self.analyzer.cleanup
                    if asyncio.iscoroutinefunction(cleanup_method):
                        await cleanup_method()
                    else:
                        cleanup_method()
            
            # Clean up UI window safely
            try:
                if cv2.getWindowProperty(self.ui.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(self.ui.window_name)
            except cv2.error:
                pass  # Window might not exist, which is fine
            
            # Clean up main window
            cv2.destroyAllWindows()
            self.ui.hide()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Basketball Shot Analyzer with Real-time UI Controls")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "gemini"],
        default="ollama",
        help="Analysis backend to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:12b-it-q4_K_M",
        help="Model name to use with Ollama backend (default: gemma3:12b-it-q4_K_M)"
    )
    parser.add_argument(
        "--ball-type",
        choices=["BASKETBALL", "TENNIS", "SOCCER", "VOLLEYBALL", "BASEBALL", "GENERIC"],
        default="BASKETBALL",
        help="Type of ball to track (default: BASKETBALL)"
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Enable optimizations for systems with <10GB VRAM"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Process every nth frame (default: 10)"
    )
    parser.add_argument(
        "--resolution-scale",
        type=float,
        default=0.75,
        help="Scale factor for frame resolution (default: 0.75)"
    )
    parser.add_argument(
        "--no-hand-tracking",
        action="store_true",
        help="Disable hand tracking"
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=30.0,
        help="Timeout for Ollama requests in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--ollama-retry-count",
        type=int,
        default=2,
        help="Number of retries for failed Ollama requests (default: 2)"
    )
    parser.add_argument(
        "--ollama-retry-delay",
        type=float,
        default=1.0,
        help="Delay between Ollama retries in seconds (default: 1.0)"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    analyzer = WebcamAnalyzerWithUI(args)
    await analyzer.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
