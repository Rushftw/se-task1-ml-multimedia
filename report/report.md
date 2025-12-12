# Software Engineering Task 1 – ML Multimedia & Local LLM

## 1. Project overview

Goal of the task:  
Use **ready-made machine learning libraries and pretrained models** to solve 4 applied tasks with different content types (text, audio, image, video) + 1 local LLM, all running **locally** on my computer (no external web services).

Repository structure:

- `text_sentiment_hf/` – text sentiment analysis (Hugging Face, Transformers)
- `audio_classification_tf/` – audio classification (TensorFlow + YAMNet)
- `image_classification_pt/` – image classification (PyTorch ResNet-18)
- `video_classification_hf/` – video classification (Hugging Face VideoMAE)
- `local_llm/` – local LLM chat using TinyLlama
- `report/` – this report
- `.gitignore` – ignores virtual environments, caches, etc.

Each task is implemented in its own folder with:

- `requirements.txt` – dependencies
- `main.py` / `chat.py` – entry script / demo
- Separate Python virtual environment (`.venv`) for isolation (not committed).

---

## 2. Used libraries and frameworks

I used several standard ML frameworks, as required:

- **Hugging Face Transformers**
  - Text sentiment analysis
  - Video classification (pipeline API)
- **TensorFlow + TensorFlow Hub + TensorFlow I/O**
  - Audio classification with YAMNet
- **PyTorch + Torchvision**
  - Image classification (ResNet-18)
- **Transformers + PyTorch**
  - Local LLM (TinyLlama)

Reasons to use them:

- All of them provide **pretrained models** for typical tasks.
- Active ecosystems, good documentation, many examples.
- Easy integration into Python scripts.
- Can run fully offline once models are downloaded.

---

## 3. Text processing – Sentiment analysis (Hugging Face)

Folder: `text_sentiment_hf/`  
Main file: `main.py`  

### 3.1. Library and advantages

- Library: **transformers** (Hugging Face)
- Advantages:
  - Very simple high-level API (`pipeline("sentiment-analysis")`)
  - Access to many pretrained NLP models
  - Handles tokenization, batching and device management internally
  - Same API can be reused later through a REST API

### 3.2. Model and principle of operation

- Model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- Architecture: **DistilBERT**, a smaller and faster version of BERT.
- Principle:
  1. Input text is tokenized into WordPiece tokens.
  2. Tokens are converted to IDs from the pretrained vocabulary.
  3. Model outputs logits for 2 classes: `POSITIVE` / `NEGATIVE`.
  4. Softmax is applied; the predicted label with highest probability is returned.

Reason for selection:

- Light-weight, so it runs fast on CPU.
- Fine-tuned specifically for **sentiment** (binary classification).
- Available directly via `pipeline`, no extra configuration.

### 3.3. Dataset structure (training / inference)

I do not train the model myself; I reuse the pretrained weights.

- Training dataset (original): **SST-2** (Stanford Sentiment Treebank)
  - Text samples: movie review sentences.
  - Label: positive / negative.
- Inference input (my app): a **single string** entered by user in console.
- Output: JSON-like dict with:
  - `label`: `"POSITIVE"` or `"NEGATIVE"`
  - `score`: probability from 0…1

### 3.4. Quality metrics

For training phase (original model):

- **Accuracy**, **Precision**, **Recall**, **F1-score** on the validation set.
- For SST-2, accuracy is high (~90%+ in publications).

For my small console application:

- Manual sanity check on test sentences.
- Product metric idea: percentage of user inputs where prediction “feels right”.

Infrastructure metric:

- Response time for one request on CPU (roughly < 0.5 s on my machine).

### 3.5. Implementation details

- File: `text_sentiment_hf/main.py`
- Uses `pipeline("sentiment-analysis", model=...)`.
- Simple interactive loop: user types a sentence, model prints label + score.
- Runs on **CPU** by default (no GPU required).
- This module is a good candidate for later wrapping into a **REST API**:
  - Endpoint: `POST /api/sentiment`
  - Body: `{ "text": "some sentence" }`
  - Response: `{ "label": "...", "score": 0.99 }`

---

## 4. Audio processing – Sound classification (TensorFlow + YAMNet)

Folder: `audio_classification_tf/`  
Main file: `main.py`  

### 4.1. Library and advantages

- Libraries: **TensorFlow**, **TensorFlow Hub**, **TensorFlow I/O**
- Advantages:
  - YAMNet is provided as a ready **TFHub** model.
  - Model is already trained on a large audio dataset (AudioSet).
  - TF audio ops work well on CPU.
  - Good documentation and examples.

### 4.2. Model and principle of operation

- Model: **YAMNet** (`https://tfhub.dev/google/yamnet/1`)
- Architecture: MobileNet-style convolutional network on top of log-mel spectrogram.
- Principle:
  1. Input: mono waveform at 16 kHz.
  2. Model converts waveform → spectrogram → embeddings.
  3. Outputs frame-level class scores for **521 sound event classes** (AudioSet ontology).
  4. I average scores over time and take Top-5 classes.

Reason for selection:

- Ready-made model for **general sound classification**.
- Works out-of-the-box on arbitrary `.wav` files.
- High coverage of many sound types (speech, music, water, etc.).

### 4.3. Dataset structure

Training dataset (original):

- **AudioSet**: clips from YouTube, each annotated with multiple sound labels.

Inference in my app:

- Input: path to a `.wav` file on disk.
- I load the file, convert to mono, resample to 16 kHz if needed.
- Output: list of Top-5 `(label, score)` pairs.

### 4.4. Quality metrics

Typical ML metrics:

- Multi-label classification metrics: **mAP**, **AUC**, etc. (published for YAMNet).
- For my usage, I focus on **Top-1** and **Top-5** accuracy “by eye” on several test samples.

Product metrics:

- Whether detected labels make sense for typical user audio (e.g., speech, water, music).
- If integrated in a product: percentage of correctly detected dominant sound.

Infrastructure metrics:

- Time to process one file.
- Memory usage while running YAMNet.

### 4.5. Implementation details

- Script `main.py`:
  - Loads YAMNet model from TFHub.
  - Loads `.wav` file passed via command line.
  - Handles sample rate conversion.
  - Computes mean scores per class and prints Top-5 classes.
- Runs on CPU; no GPU required for short clips.
- The function could later be wrapped into an API endpoint:
  - `POST /api/audio-classification` with audio file upload.

---

## 5. Image processing – Image classification (PyTorch ResNet-18)

Folder: `image_classification_pt/`  
Main file: `main.py`  

### 5.1. Library and advantages

- Libraries: **PyTorch**, **torchvision**
- Advantages:
  - `torchvision.models` exposes many pretrained CNNs on ImageNet.
  - Loading weights is one line (`models.resnet18(weights=...)`).
  - High performance; widely used in research/industry.

### 5.2. Model and principle of operation

- Model: **ResNet-18** pretrained on ImageNet.
- Architecture: convolutional neural network with residual connections.
- Principle:
  1. Input image is resized to 256×256 and center-cropped to 224×224.
  2. Normalized with ImageNet mean/std.
  3. Forward pass outputs logits for **1000 ImageNet classes**.
  4. Apply softmax and take Top-5 highest scores.

Reason for selection:

- Classic, lightweight model.
- Fast to run on CPU.
- Good enough accuracy for demo and experimentation.

### 5.3. Dataset structure

Training dataset (original):

- **ImageNet-1K**:
  - 1.2M training images.
  - 1000 classes (dog breeds, objects, etc.).

Inference in my app:

- Input: path to an image file (JPG/PNG).
- Output: Top-5 classes with scores.
- I download ImageNet class labels from GitHub as a text file.

### 5.4. Quality metrics

For training (original):

- **Top-1** and **Top-5** accuracy on ImageNet validation set.

For my simple app:

- Manual check of predictions on few pictures (e.g. dog, car).
- Product metric idea: fraction of cases where Top-5 list contains the correct label.

Infrastructure metrics:

- Inference latency on CPU.
- If extended, number of images processed per second.

### 5.5. Implementation details

- `main.py`:
  - Loads ResNet-18 with pretrained weights.
  - Uses `torchvision.transforms` for preprocessing.
  - Performs softmax and prints Top-5 predictions.
- Clean separation between:
  - Loading labels
  - Preprocessing
  - Model inference

---

## 6. Video processing – Video classification (Hugging Face VideoMAE)

Folder: `video_classification_hf/`  
Main file: `main.py`  

### 6.1. Library and advantages

- Library: **transformers** with **video-classification pipeline**
- Advantages:
  - Same high-level `pipeline` interface as text.
  - Support for video models like VideoMAE.
  - Handles reading video, sampling frames, etc.

### 6.2. Model and principle of operation

- Model: `MCG-NJU/videomae-base-finetuned-kinetics`
- Architecture: **VideoMAE** – masked autoencoder applied to video frames.
- Principle:
  1. Video is decoded into frames (subsampled).
  2. Frames are passed through a transformer encoder.
  3. Final embedding is used for action classification over Kinetics classes.

Reason for selection:

- Ready HF model for generic **human action recognition**.
- One-line usage with `pipeline("video-classification", model=...)`.

### 6.3. Dataset structure

Training dataset (original):

- **Kinetics** (e.g. Kinetics-400/600/700):
  - Short video clips with labels describing actions (“playing guitar”, “running”, etc.).

Inference in my app:

- Input: path to a local video file (`.mp4`).
- Output: list of `(label, score)` predictions.

### 6.4. Quality metrics

ML metrics (original model):

- Top-1 / Top-5 accuracy on Kinetics validation set.

Product metrics:

- For an application like video-tagging:
  - Percentage of videos where predicted tags are useful.
  - Tag coverage (how many useful labels per video).

Infrastructure metrics:

- Inference time for a N-second clip.
- Throughput if used in batch processing.

### 6.5. Implementation details

- `main.py`:
  - Validates file path from command line.
  - Creates `pipeline("video-classification", model=MODEL_NAME)`.
  - Calls it on the video path and prints returned labels + scores.
- Video loading and frame sampling are handled internally by the pipeline.

---

## 7. Local LLM – TinyLlama Chat

Folder: `local_llm/`  
Main file: `chat.py`  

### 7.1. Library and advantages

- Libraries: **transformers**, **torch**, **accelerate**
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Advantages:
  - Relatively small model (~1.1B parameters) → possible to run on CPU.
  - Open source; can be deployed fully **locally**, no third-party API.
  - Compatible with standard causal LM generation API.

### 7.2. Model and principle of operation

- Architecture: Decoder-only transformer (GPT-style).
- Principle:
  1. User enters a prompt in console.
  2. Prompt is tokenized to IDs.
  3. Model generates new tokens autoregressively (`generate`).
  4. Tokens are decoded back to text and printed.

Reason for selection:

- Fits in local resources.
- Chat-finetuned model → gives usable conversational answers for demo.

### 7.3. Dataset structure

Training data (original):

- Mixture of text corpora + conversational data (as published by model authors).

Inference in my app:

- Input: free-form prompt string.
- Output: generated continuation text (approx up to 120 tokens).

### 7.4. Quality metrics

Possible evaluation metrics:

- Perplexity on held-out text (for language modelling).
- Human evaluation of:
  - Relevance
  - Fluency
  - Response time

Infrastructure metrics:

- Time to generate N tokens on CPU.
- Peak RAM usage.

### 7.5. Implementation details

- `chat.py`:
  - Loads tokenizer + model from Hugging Face.
  - Simple REPL loop:
    - `You: ...`
    - `Assistant: ...`
  - Uses `model.generate` with `max_new_tokens`, `temperature`, `do_sample=True`.
- Completely local:
  - After first download, it does not require internet access.

---

## 8. Potential API design (for later steps)

Task description mentions that one of the models will later be exposed through an API.  
The most natural candidate is **text sentiment analysis**:

Example API design:

- **Endpoint:** `POST /api/v1/sentiment`
- **Request body:**
  ```json
  {
    "text": "This course is great!"
  }
