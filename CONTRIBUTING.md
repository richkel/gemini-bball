# Contributing to Basketball Shot Analyzer

Thank you for your interest in contributing to the Basketball Shot Analyzer project! We welcome contributions from the community.

## Setting Up the Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/basketball-shot-analyzer.git
   cd basketball-shot-analyzer
   ```

3. **Set up a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

## Running Tests

All test scripts are organized in the `tests/` directory. You can run individual tests or the entire test suite:

```bash
# Run installation verification
python -m tests.test_installation

# Test core functionality
python -m tests.test_core

# Test the Ollama backend
python -m tests.test_ollama_detector --image path/to/image.jpg

# Test with URL images
python -m tests.test_ollama_url --url https://example.com/image.jpg

# Test multi-ball detection
python -m tests.test_multi_ball_detection --ball-type BASKETBALL --use-ollama

# Test webcam functionality
python -m tests.test_webcam

# Test real-time analysis
python -m tests.test_realtime

# Test Gemini backend (requires API key)
python -m tests.test_gemini
```

When adding new tests, please place them in the `tests/` directory and follow the existing naming convention.

## Code Style

Please follow these guidelines when contributing code:

- Use 4 spaces for indentation (PEP 8)
- Follow PEP 8 style guide
- Write docstrings for all public functions and classes
- Add type hints for better code clarity
- Keep lines under 100 characters when possible

## Submitting Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. Make your changes and commit them with a clear message:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

3. Push your changes to your fork:
   ```bash
   git push origin your-branch-name
   ```

4. Open a pull request against the main branch of the original repository.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear title
- Detailed description
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Screenshots or logs if applicable

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.
