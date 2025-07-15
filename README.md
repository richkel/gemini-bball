# Basketball Shot Analyzer

A real-time basketball shot analysis tool using computer vision and AI. Analyze shooting form, track shot accuracy, and get performance metrics in real-time or from recorded videos.

![Basketball Shot Analyzer Demo](demo.gif)

## ✨ Features

- 🎯 Real-time shot detection and analysis
- 🤖 Multiple AI backends (Ollama, Gemini)
- 🏀 Multi-ball detection (basketball, soccer, volleyball, tennis, baseball)
- 📊 Shot statistics and performance metrics
- 🎥 Video recording and playback
- 📈 Performance tracking over time
- 🛠️ Customizable analysis parameters

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for real-time analysis)
- [Ollama](https://ollama.ai/) installed (for local analysis)
- Google API key (for Gemini backend)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/richkel/gemini-bball.git
   cd gemini-bball
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
   pip install -e .  # Install in development mode
   ```

4. Set up Ollama (for local analysis):
   ```bash
   # Start Ollama server
   ollama serve
   
   # Pull the recommended model
   ollama pull gemma3:12b-it-q4_K_M
   ```

5. Set up Gemini API (optional, for Gemini backend):
   ```bash
   # Windows
   set GOOGLE_API_KEY=your_api_key_here
   
   # macOS/Linux
   export GOOGLE_API_KEY=your_api_key_here
   ```

### Basic Usage

#### Local Analysis with Ollama

```bash
# Run with default settings (Ollama backend)
python -m src.cli

# Analyze a video file
python -m src.cli --source path/to/your/video.mp4 --backend ollama

# Save analysis to file
python -m src.cli --output analysis_results.mp4 --backend ollama

# Specify a different Ollama model
python -m src.cli --backend ollama --model "llava:latest"
```

#### Cloud Analysis with Gemini API

```bash
# Use Gemini backend (requires API key)
python -m src.cli --backend gemini

# Analyze a video with Gemini backend
python -m src.cli --source path/to/your/video.mp4 --backend gemini

# Save Gemini analysis to file
python -m src.cli --output gemini_analysis.mp4 --backend gemini
```

#### Multi-Ball Detection

```bash
# Run with multi-ball detection demo (basketball)
python run_multi_ball_detector.py --ball-type BASKETBALL --use-ollama

# Switch to a different ball type
python run_multi_ball_detector.py --ball-type SOCCER --use-ollama

# Run without Ollama (color-based tracking only)
python run_multi_ball_detector.py --ball-type VOLLEYBALL
```

#### Runtime Controls

When running the application:
- Press `q` to quit
- Press `b` to cycle through ball types (in multi-ball detector)
- Press `o` to toggle Ollama detection on/off (in multi-ball detector)
- Press `s` to save the current frame (in multi-ball detector)

## 📚 Documentation

For detailed instructions, configuration options, and troubleshooting, please see the [User Guide](USER_GUIDE.md).

## 👨‍💻 API Integration

### Gemini API

To use the Gemini API backend:

1. Get an API key from [Google AI Studio](https://ai.google.dev/)
2. Set the API key as an environment variable:
   ```bash
   # Windows
   set GOOGLE_API_KEY=your_api_key_here
   
   # macOS/Linux
   export GOOGLE_API_KEY=your_api_key_here
   ```
3. Run the application with the Gemini backend:
   ```bash
   python -m src.cli --backend gemini
   ```

### Ollama API

The application uses Ollama's local API by default:

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```
2. Check available models:
   ```bash
   ollama list
   ```
3. Run with a specific model:
   ```bash
   python -m src.cli --backend ollama --model "gemma3:12b-it-q4_K_M"
   ```

### Custom API Endpoints

You can configure custom API endpoints through environment variables:

```bash
# Windows
set OLLAMA_HOST=http://custom-ollama-server:11434

# macOS/Linux
export OLLAMA_HOST=http://custom-ollama-server:11434
```

## 🔥 GitHub Repository

This project is available on GitHub at [https://github.com/richkel/gemini-bball](https://github.com/richkel/gemini-bball)

## 📊 Example Output

```
2025-07-07 15:30:45,123 - Shot detected: JUMP_SHOT - MADE (Confidence: 0.92)
2025-07-07 15:30:52,789 - Shot detected: THREE_POINTER - MISSED (Confidence: 0.87)
2025-07-07 15:31:00,000 - Analysis complete: 5 shots, 3 made (60.0%)
```

## 🧪 Testing

The project includes a comprehensive test suite in the `tests/` directory:

```bash
# Run a specific test
python -m tests.test_core

# Test the Ollama object detector
python -m tests.test_ollama_detector --image path/to/image.jpg

# Test multi-ball detection
python -m tests.test_multi_ball_detection --ball-type BASKETBALL --use-ollama

# Test webcam functionality
python -m tests.test_webcam
```

For more information about testing, see the [Contributing Guidelines](CONTRIBUTING.md).

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
