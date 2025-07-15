from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="basketball_shot_analyzer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "pillow>=10.0.0",
        "httpx>=0.24.0",
        "aiohttp>=3.8.4",
        "google-generativeai>=0.3.0",
        "protobuf>=4.25.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Basketball Shot Analyzer using Computer Vision and AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/basketball-shot-analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "basketball-analyzer=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "basketball_shot_analyzer": ["*.json", "*.yaml"],
    },
)
