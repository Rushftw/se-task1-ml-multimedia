# Software Engineering Task 1: ML Multimedia & Local LLM

This project demonstrates the application of pre-trained machine learning models for text sentiment analysis, audio classification, image classification, and video classification. Additionally, a Local Large Language Model (LLM) is used for text generation.

## 1. Project Setup

### Prerequisites:
- Python 3.11+
- Git
- An internet connection (for initial downloads)

### Setting up the Environment:

1. **Clone the repository:**

   First, clone the repository to your local machine using Git:

   ```bash
   git clone https://github.com/Rushftw/se-task1-ml-multimedia.git
   cd se-task1-ml-multimedia

2. ## Create a virtual environment:

Use Python's built-in venv module to create a virtual environment:
    
    python -m venv .venv
    
    
## Activate the virtual environment:

On Windows (PowerShell or CMD):     .venv\Scripts\activate


## Install dependencies:

With the virtual environment activated, install the required libraries:     
                                                                         pip install -r requirements.txt


2. Running the Tasks
## Task 1: Text Sentiment Analysis

Run the sentiment analysis model on the text you provide:     python text_sentiment_hf/main.py


## Task 2: Audio Classification

Classify audio files into various categories:

python audio_classification_tf/main.py



## Task 3: Image Classification

Classify images using a pre-trained image classification model:

python image_classification_pt/main.py


## Task 4: Video Classification

Classify video files with the pre-trained video classification model:

python video_classification_hf/main.py

## Task 5: Local LLM (Tiny Llama Chat)

Run the Local Large Language Model for chat-like interactions:

python local_llm/chat.py

## 3. Report

The report.md file contains a detailed explanation of the tasks and results.

It also includes API design for future steps (exposing models through an API).

## 5. Contact & License

Feel free to reach out if you have any questions or encounter issues!

GitHub: https://github.com/Rushftw/se-task1-ml-multimedia