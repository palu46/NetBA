# NetBA

This project focuses on player tracking and visual analysis in NBA basketball games using computer vision techniques. 

The main objectives are:

- **Player Detection & Tracking**: Identify and follow the players on the court throughout the game footage, even when the camera angle changes.
- **Homographic Court Transformation**: Generate a bird's eye view of the court (top-down perspective) in real-time to visualize player positioning more intuitively.
- **Statistical Analysis**: Analyze gameplay dynamics and extract meaningful statistics such as player movement patterns, heatmaps, and positioning metrics.

The project leverages image rectification, homography estimation, and tracking algorithms to achieve these goals.

## Installation

Before installing the project dependencies, make sure to install **PaddlePaddle**, which is required for OCR functionality.

1. **Install PaddlePaddle**

   Follow the official installation guide based on your system and environment:

   👉 [PaddlePaddle Installation Guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=undefined)

2. **Install project dependencies**

   After PaddlePaddle is installed, you can install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Project

To run the project, simply call `main.py` with the path to the input video as a command-line argument:

```bash
python main.py path/to/your/video.mp4
```

The `data/` directory already contains some example video files you can use for testing.

### 🎬 Output

The program produces two annotated videos:
- `players_detection.mp4`
- `radar.mp4`

These will be saved in the `output/` directory after processing.

## Model Files & Git LFS

This project includes large model files (e.g., YOLO and PaddleOCR), which are managed using **Git LFS**.

### 🛠️ Cloning the Repository

If you're cloning this repository, make sure to install Git LFS **before** working with the model files:

1. **Install Git LFS**

   - **Ubuntu/Debian:** `sudo apt install git-lfs`
   - **macOS (Homebrew):** `brew install git-lfs`
   - **Windows:** https://git-lfs.github.com

2. **Initialize Git LFS**

   After cloning the repo, run:

   ```bash
   git lfs install
   ```

This ensures all large model files are properly downloaded. Otherwise, you may see placeholder text files instead of actual models.

## ⚡ Optional: Speed Up Builds with ccache

Installing `ccache` can significantly reduce build times.

### Quick Install

- **Ubuntu/Debian:** `sudo apt install ccache`
- **macOS (Homebrew):** `brew install ccache`

You can safely ignore any warnings if it's not installed, but it's recommended for faster repeated builds.