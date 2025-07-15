"""
UI Controls for Basketball Shot Analyzer

This module provides a simple UI for adjusting settings in real-time.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Callable, Optional, List, Tuple

logger = logging.getLogger(__name__)

class SettingsUI:
    """Simple UI for adjusting settings in real-time."""
    
    def __init__(self, window_name: str = "Settings"):
        """Initialize the settings UI.
        
        Args:
            window_name: Name of the settings window
        """
        self.window_name = window_name
        self.settings: Dict[str, Any] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.ui_height = 400
        self.ui_width = 300
        self.padding = 10
        self.slider_height = 30
        self.button_height = 40
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.text_color = (255, 255, 255)
        self.bg_color = (60, 60, 60)
        self.slider_color = (0, 120, 255)
        self.button_color = (0, 180, 0)
        self.button_hover_color = (0, 220, 0)
        self.button_active_color = (0, 255, 0)
        self.slider_positions: Dict[str, int] = {}
        self.toggle_states: Dict[str, bool] = {}
        self.mouse_pos = (0, 0)
        self.active_slider: Optional[str] = None
        self.visible = False
        self.ui_canvas = np.zeros((self.ui_height, self.ui_width, 3), dtype=np.uint8)
        
    def add_slider(self, name: str, min_val: float, max_val: float, 
                  default_val: float, callback: Optional[Callable] = None):
        """Add a slider control.
        
        Args:
            name: Name of the slider
            min_val: Minimum value
            max_val: Maximum value
            default_val: Default value
            callback: Function to call when slider value changes
        """
        self.settings[name] = {
            "type": "slider",
            "min": min_val,
            "max": max_val,
            "value": default_val,
            "display_name": name.replace("_", " ").title()
        }
        if callback:
            self.callbacks[name] = callback
        self.slider_positions[name] = int((default_val - min_val) / (max_val - min_val) * (self.ui_width - 2 * self.padding))
    
    def add_toggle(self, name: str, default_state: bool, callback: Optional[Callable] = None):
        """Add a toggle button.
        
        Args:
            name: Name of the toggle
            default_state: Default state (True/False)
            callback: Function to call when toggle state changes
        """
        self.settings[name] = {
            "type": "toggle",
            "value": default_state,
            "display_name": name.replace("_", " ").title()
        }
        if callback:
            self.callbacks[name] = callback
        self.toggle_states[name] = default_state
    
    def add_button(self, name: str, callback: Optional[Callable] = None):
        """Add a button.
        
        Args:
            name: Name of the button
            callback: Function to call when button is clicked
        """
        self.settings[name] = {
            "type": "button",
            "display_name": name.replace("_", " ").title()
        }
        if callback:
            self.callbacks[name] = callback
    
    def show(self):
        """Show the settings UI."""
        if not self.visible:
            self.visible = True
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            self._update_ui()
            logger.info("Settings UI opened")
    
    def hide(self):
        """Hide the settings UI."""
        if self.visible:
            self.visible = False
            cv2.destroyWindow(self.window_name)
            logger.info("Settings UI closed")
    
    def toggle_visibility(self):
        """Toggle UI visibility."""
        if self.visible:
            self.hide()
        else:
            self.show()
    
    def get_value(self, name: str) -> Any:
        """Get the current value of a setting.
        
        Args:
            name: Name of the setting
            
        Returns:
            Current value of the setting
        """
        if name in self.settings:
            return self.settings[name]["value"]
        return None
    
    def set_value(self, name: str, value: Any):
        """Set the value of a setting.
        
        Args:
            name: Name of the setting
            value: New value
        """
        if name in self.settings:
            self.settings[name]["value"] = value
            if self.settings[name]["type"] == "slider":
                min_val = self.settings[name]["min"]
                max_val = self.settings[name]["max"]
                self.slider_positions[name] = int((value - min_val) / (max_val - min_val) * (self.ui_width - 2 * self.padding))
            elif self.settings[name]["type"] == "toggle":
                self.toggle_states[name] = value
            
            # Call the callback if it exists
            if name in self.callbacks:
                self.callbacks[name](value)
            
            if self.visible:
                self._update_ui()
    
    def update(self):
        """Update the UI if visible."""
        if self.visible:
            self._update_ui()
            cv2.imshow(self.window_name, self.ui_canvas)
            cv2.waitKey(1)
    
    def _update_ui(self):
        """Update the UI canvas."""
        # Create background
        self.ui_canvas = np.full((self.ui_height, self.ui_width, 3), self.bg_color, dtype=np.uint8)
        
        # Draw title
        cv2.putText(self.ui_canvas, "Settings", (self.padding, 30), 
                   self.font, 1.0, self.text_color, 2)
        
        y_pos = 60
        
        # Draw controls
        for name, setting in self.settings.items():
            display_name = setting["display_name"]
            
            if setting["type"] == "slider":
                # Draw slider label and value
                value_text = f"{display_name}: {setting['value']:.2f}"
                cv2.putText(self.ui_canvas, value_text, 
                           (self.padding, y_pos), self.font, self.font_scale, 
                           self.text_color, 1)
                
                # Draw slider track
                track_y = y_pos + 15
                cv2.rectangle(self.ui_canvas, 
                             (self.padding, track_y), 
                             (self.ui_width - self.padding, track_y + 10), 
                             (100, 100, 100), -1)
                
                # Draw slider handle
                handle_x = self.padding + self.slider_positions[name]
                cv2.rectangle(self.ui_canvas, 
                             (handle_x - 5, track_y - 5), 
                             (handle_x + 5, track_y + 15), 
                             self.slider_color, -1)
                
                y_pos += self.slider_height + 10
                
            elif setting["type"] == "toggle":
                # Draw toggle label
                cv2.putText(self.ui_canvas, display_name, 
                           (self.padding, y_pos), self.font, self.font_scale, 
                           self.text_color, 1)
                
                # Draw toggle button
                toggle_x = self.ui_width - 60
                toggle_y = y_pos - 15
                toggle_color = self.button_active_color if self.toggle_states[name] else (100, 100, 100)
                cv2.rectangle(self.ui_canvas, 
                             (toggle_x, toggle_y), 
                             (toggle_x + 40, toggle_y + 20), 
                             toggle_color, -1)
                
                # Draw toggle state text
                state_text = "ON" if self.toggle_states[name] else "OFF"
                text_x = toggle_x + 10
                cv2.putText(self.ui_canvas, state_text, 
                           (text_x, toggle_y + 15), self.font, 0.5, 
                           (0, 0, 0), 1)
                
                y_pos += self.button_height
                
            elif setting["type"] == "button":
                # Draw button
                button_y = y_pos - 15
                cv2.rectangle(self.ui_canvas, 
                             (self.padding, button_y), 
                             (self.ui_width - self.padding, button_y + 30), 
                             self.button_color, -1)
                
                # Draw button text
                text_size = cv2.getTextSize(display_name, self.font, self.font_scale, 1)[0]
                text_x = self.padding + (self.ui_width - 2 * self.padding - text_size[0]) // 2
                cv2.putText(self.ui_canvas, display_name, 
                           (text_x, button_y + 20), self.font, self.font_scale, 
                           (0, 0, 0), 1)
                
                y_pos += self.button_height + 10
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events.
        
        Args:
            event: Mouse event type
            x: X coordinate
            y: Y coordinate
            flags: Event flags
            param: Additional parameters
        """
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked on a slider
            for name, setting in self.settings.items():
                if setting["type"] == "slider":
                    slider_y = self._get_slider_y_position(name)
                    if (y >= slider_y and y <= slider_y + 10 and 
                        x >= self.padding and x <= self.ui_width - self.padding):
                        self.active_slider = name
                        self._update_slider_position(name, x)
                        break
            
            # Check if clicked on a toggle
            for name, setting in self.settings.items():
                if setting["type"] == "toggle":
                    toggle_y = self._get_toggle_y_position(name)
                    toggle_x = self.ui_width - 60
                    if (y >= toggle_y and y <= toggle_y + 20 and 
                        x >= toggle_x and x <= toggle_x + 40):
                        self.toggle_states[name] = not self.toggle_states[name]
                        self.settings[name]["value"] = self.toggle_states[name]
                        if name in self.callbacks:
                            self.callbacks[name](self.toggle_states[name])
                        self._update_ui()
                        break
            
            # Check if clicked on a button
            for name, setting in self.settings.items():
                if setting["type"] == "button":
                    button_y = self._get_button_y_position(name)
                    if (y >= button_y and y <= button_y + 30 and 
                        x >= self.padding and x <= self.ui_width - self.padding):
                        if name in self.callbacks:
                            self.callbacks[name]()
                        break
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.active_slider = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.active_slider:
                self._update_slider_position(self.active_slider, x)
    
    def _update_slider_position(self, name: str, x: int):
        """Update slider position and value.
        
        Args:
            name: Name of the slider
            x: X coordinate
        """
        # Clamp x position to slider track
        x = max(self.padding, min(x, self.ui_width - self.padding))
        
        # Update slider position
        self.slider_positions[name] = x - self.padding
        
        # Update setting value
        min_val = self.settings[name]["min"]
        max_val = self.settings[name]["max"]
        normalized_pos = self.slider_positions[name] / (self.ui_width - 2 * self.padding)
        value = min_val + normalized_pos * (max_val - min_val)
        self.settings[name]["value"] = value
        
        # Call callback if it exists
        if name in self.callbacks:
            self.callbacks[name](value)
        
        self._update_ui()
    
    def _get_slider_y_position(self, name: str) -> int:
        """Get the Y position of a slider.
        
        Args:
            name: Name of the slider
            
        Returns:
            Y position of the slider
        """
        y_pos = 60
        for n, s in self.settings.items():
            if n == name:
                return y_pos + 15
            if s["type"] == "slider":
                y_pos += self.slider_height + 10
            elif s["type"] == "toggle":
                y_pos += self.button_height
            elif s["type"] == "button":
                y_pos += self.button_height + 10
        return 0
    
    def _get_toggle_y_position(self, name: str) -> int:
        """Get the Y position of a toggle.
        
        Args:
            name: Name of the toggle
            
        Returns:
            Y position of the toggle
        """
        y_pos = 60
        for n, s in self.settings.items():
            if n == name:
                return y_pos - 15
            if s["type"] == "slider":
                y_pos += self.slider_height + 10
            elif s["type"] == "toggle":
                y_pos += self.button_height
            elif s["type"] == "button":
                y_pos += self.button_height + 10
        return 0
    
    def _get_button_y_position(self, name: str) -> int:
        """Get the Y position of a button.
        
        Args:
            name: Name of the button
            
        Returns:
            Y position of the button
        """
        y_pos = 60
        for n, s in self.settings.items():
            if n == name:
                return y_pos - 15
            if s["type"] == "slider":
                y_pos += self.slider_height + 10
            elif s["type"] == "toggle":
                y_pos += self.button_height
            elif s["type"] == "button":
                y_pos += self.button_height + 10
        return 0


def create_default_ui() -> SettingsUI:
    """Create a default settings UI with common controls.
    
    Returns:
        Configured SettingsUI instance
    """
    ui = SettingsUI()
    
    # Add performance controls
    ui.add_toggle("low_vram_mode", True)
    ui.add_slider("frame_skip", 1, 30, 5)
    ui.add_slider("resolution_scale", 0.25, 1.0, 1.0)
    ui.add_toggle("enable_caching", True)
    
    # Add tracking controls
    ui.add_toggle("hand_tracking", True)
    ui.add_toggle("ball_tracking", True)
    ui.add_toggle("use_ollama_detection", False)
    
    # Add action buttons
    ui.add_button("save_settings")
    ui.add_button("reset_defaults")
    
    return ui
