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