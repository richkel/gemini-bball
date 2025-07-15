# Basketball Shot Analyzer - User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Configuration Options](#configuration-options)
5. [Use Cases](#use-cases)
   - [Real-time Webcam Analysis](#1-real-time-webcam-analysis)
   - [Video File Analysis](#2-video-file-analysis)
   - [Batch Processing](#3-batch-processing)
6. [Troubleshooting](#troubleshooting)
7. [Example Log Output](#example-log-output)
8. [Advanced Configuration](#advanced-configuration)

## Quick Start

1. Install the package:
   ```bash
   pip install -r requirements.txt
   ```

2. Run with default settings (webcam):
   ```bash
   python -m src.cli
   ```

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time analysis)
- NVIDIA GPU recommended for better performance

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/basketball-shot-analyzer.git
   cd basketball-shot-analyzer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### Running the Analyzer
```bash
# Basic usage with webcam (press 'q' to quit)
python -m src.cli

# Analyze a video file
python -m src.cli --source path/to/video.mp4

# Save output to file
python -m src.cli --output analysis_output.mp4

# Use Gemini backend (requires API key)
python -m src.cli --backend gemini --api-key YOUR_API_KEY
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Video source (camera index or file path) | `0` (webcam) |
| `--backend` | Analysis backend (`ollama` or `gemini`) | `ollama` |
| `--model` | Model to use for analysis | `llava:latest` |
| `--api-key` | API key (required for Gemini) | `None` |
| `--frame-skip` | Process every N frames | `5` |
| `--output` | Output file path | `output.mp4` |
| `--debug` | Enable debug logging | `False` |
| `--ball-type` | Type of ball to track | `BASKETBALL` |
| `--use-ollama` | Use Ollama for object detection | `False` |

## Use Cases

### 1. Real-time Webcam Analysis
```bash
# Basic real-time analysis
python -m src.cli

# Higher processing rate (lower frame-skip)
python -m src.cli --frame-skip 2

# Multi-ball detection with Ollama
python run_multi_ball_detector.py --use-ollama
```

### 2. Video File Analysis
```bash
# Analyze a video file
python -m src.cli --source game_recording.mp4

# Save analysis to file
python -m src.cli --source game_recording.mp4 --output analysis_results.mp4
```

### 3. Batch Processing
```bash
# Process multiple videos
for video in games/*.mp4; do
    python -m src.cli --source "$video" --output "analysis/${video##*/}"
done
```

## Troubleshooting

### Common Issues

#### 1. Webcam Not Detected
- **Symptom**: `Could not open video source`
- **Solution**:
  - Check if the webcam is connected
  - Try a different camera index: `--source 1`
  - On Linux, ensure proper permissions: `sudo usermod -a -G video $USER`

#### 2. Model Loading Issues
- **Symptom**: `Error loading model`
- **Solution**:
  - Ensure Ollama is running: `ollama serve`
  - Pull the required model: `ollama pull gemma3:12b-it-q4_K_M`
  - Check available models: `ollama list`

#### 3. Performance Problems
- **Symptom**: Low FPS or laggy performance
- **Solution**:
  - Increase frame-skip: `--frame-skip 10`
  - Reduce resolution in your camera settings
  - Use a more powerful GPU

#### 4. API Key Issues (Gemini)
- **Symptom**: `Invalid API key`
- **Solution**:
  - Get a valid API key from Google AI Studio
  - Set it via command line or environment variable:
    ```bash
    export GOOGLE_API_KEY=your_api_key
    python -m src.cli --backend gemini
    ```

## Example Log Output

```
2025-07-07 15:30:45,123 - src.pipeline.real_time_analyzer - INFO - Starting real-time analysis (Press 'q' to quit)
2025-07-07 15:30:47,567 - src.analysis.ollama_backend - INFO - Initialized Ollama backend with model: llava:latest
2025-07-07 15:30:50,234 - src.pipeline.real_time_analyzer - INFO - Shot detected: JUMP_SHOT - MADE (Confidence: 0.92)
2025-07-07 15:30:52,789 - src.pipeline.real_time_analyzer - INFO - Shot detected: THREE_POINTER - MISSED (Confidence: 0.87)
2025-07-07 15:30:55,456 - src.pipeline.real_time_analyzer - INFO - Shot detected: LAYUP - MADE (Confidence: 0.94)
2025-07-07 15:30:58,123 - src.pipeline.real_time_analyzer - INFO - Statistics: 3 total shots, 2 made (66.7%)
2025-07-07 15:31:00,000 - src.pipeline.real_time_analyzer - INFO - Analysis completed. Results saved to analysis.json
```

## Advanced Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | API key for Gemini backend | `None` |
| `OLLAMA_HOST` | Custom Ollama server URL | `http://localhost:11434` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

### Multi-Ball Detection

The analyzer supports tracking different types of sports balls:

```bash
# Run the multi-ball detector with a specific ball type
python run_multi_ball_detector.py --ball-type BASKETBALL

# Available ball types: BASKETBALL, SOCCER, VOLLEYBALL, TENNIS, BASEBALL, GENERIC
python run_multi_ball_detector.py --ball-type SOCCER --use-ollama
```

During runtime, you can use these keyboard controls:
- Press 'b' to cycle through different ball types
- Press 'o' to toggle Ollama detection on/off
- Press 's' to save the current frame
- Press 'q' to quit

### Custom Models

To use a different model with Ollama:

1. Pull the desired model:
   ```bash
   ollama pull your-model:tag
   ```

2. Run with the custom model:
   ```bash
   python -m src.cli --model your-model:tag
   ```

### Performance Tuning

For better performance on lower-end systems:

```bash
# Reduce processing load
python -m src.cli --frame-skip 10 --model tiny-llava:latest

# Lower resolution (if using webcam)
python -m src.cli --resolution 640x480
```

### Output Formats

The analyzer can export results in different formats:

```bash
# JSON output
python -m src.cli --output analysis.json --format json

# CSV output
python -m src.cli --output stats.csv --format csv
```

## Support

For additional help, please open an issue on our [GitHub repository](https://github.com/yourusername/basketball-shot-analyzer/issues).

---
*Last updated: July 7, 2025*
