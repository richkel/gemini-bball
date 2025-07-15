#!/usr/bin/env python
"""
Download test images for ball detection testing.
"""

import os
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create test_images directory if it doesn't exist
os.makedirs("test_images", exist_ok=True)

# URLs for different ball types
test_images = {
    "basketball.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/7a/Basketball.png",
    "soccer_ball.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Soccerball.svg/1200px-Soccerball.svg.png",
    "tennis_ball.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Tennis_ball.svg/800px-Tennis_ball.svg.png",
    "volleyball.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Volleyball.svg/800px-Volleyball.svg.png",
    "baseball.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Baseball.svg/1200px-Baseball.svg.png"
}

def download_image(url, filename):
    """Download an image from a URL and save it to a file."""
    try:
        logger.info(f"Downloading {filename} from {url}")
        
        # Set a proper User-Agent header to comply with Wikimedia's policy
        headers = {
            'User-Agent': 'GeminiBballProject/1.0 (https://github.com/user/gemini-bball; user@example.com) Python/3.x requests/2.x'
        }
        
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        with open(os.path.join("test_images", filename), 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False

def main():
    """Main function."""
    success_count = 0
    for filename, url in test_images.items():
        if download_image(url, filename):
            success_count += 1
    
    logger.info(f"Downloaded {success_count}/{len(test_images)} images")

if __name__ == "__main__":
    main()
