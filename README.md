# Video Summarization with Audio Output

Welcome to the **Video Summarization with Audio Output** project! This project summarizes videos by generating captions and converting them into audio using a deep learning pipeline. A Streamlit-powered web app allows users to upload videos, view generated captions, and listen to audio summaries.

---

## Features
- Summarizes videos into concise text captions.
- Converts captions into audio using Google Text-to-Speech (gTTS).
- Provides a web-based interface for easy interaction.
- Uses pretrained VGG16 for feature extraction from video frames.
- Handles large datasets like MSR-VTT.

---

## Requirements
- Python 3.8+
- Libraries: 
  - PyTorch
  - torchvision
  - transformers
  - gTTS
  - Streamlit
  - OpenCV

---

## Installation and Setup

Follow these steps to set up and run the project:

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/video-summarization.git
cd video-summarization
