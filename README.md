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
git clone https://github.com/b0808/non-audio-video-summarization-through-audio.git
cd non-audio-video-summarization-through-audio
```

### Step 2: Install Dependencies
Install the required libraries using pip:
```bash
pip install torch torchvision transformers gtts streamlit opencv-python-headless
```
### Step 3: Dataset Preparation
Download the MSR-VTT Dataset
Access the dataset from the official website or download through kaggle [MSRVTT](https://https://www.kaggle.com/datasets/vishnutheepb/msrvtt)
```bash
pip install torch torchvision transformers gtts streamlit opencv-python-headless
```
### Step 4: How to Run
Step 1: Preprocess the Dataset
Prepare the dataset for training by running
```bash
python preprocess.py
```
Step 2: Extract Features
Extract video frame features using VGG16:
```bash
python extract_features.py
```
Step 3: Train the Model
Train the encoder-decoder model
```bash
python fork-of-train (2).py
```
### Kaggle Integration
Prefer running the project on Kaggle? Use these notebooks:
Prepare the dataset for training by running
Check out the [Train Notebook](https://www.kaggle.com/code/bhaveshsandbhor/fork-of-train)
Check out the [Preprocessing Notebook](https://www.kaggle.com/code/bhaveshsandbhor/preprocess) 
Check out the [Feature Extraction Script](https://www.kaggle.com/code/bhaveshsandbhor/extract-features)
Check out the [Configuration File](https://www.kaggle.com/code/bhaveshsandbhor/config) 
dataset are integrated with notebook
Upload notebook to Kaggle and run the code in the respective notebooks.

###  Using the Interface
```bash
streamlit run steremlite.py
```
Steps:
1 Upload a video file.
2 View the generated caption.
3 Play the audio summary directly in the web-app.
### Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for suggestions or improvements.
