# Emotional Speech Dataset Builder

This project builds a **structured emotional speech dataset** from **YouTube videos**.  
It automatically downloads audio, extracts speech segments, transcribes them, detects speakers, and analyzes **sentiment and emotion** using multiple models (Czech Roberta, local LLM via Ollama, and optionally Wav2Vec).  
The output is a dataset organized by emotion categories.

---

## Overview

**Input:**  
- One or more YouTube URLs (listed in `urls.txt`)

**Output:**  
- Dataset in `dataset/<video_id>/<emotion|no-emotion>/<video_id>/{video_id_xx.mp3,video_id_xx.json}`
- Metadata JSON describing detected segments, speakers, transcriptions, and model scores

```
dataset
|── 5NsJZMbU9U0
│   ├── emotion
│   │   ├── 5NsJZMbU9U0_002
│   │   │   ├── 5NsJZMbU9U0_002.json
│   │   │   └── 5NsJZMbU9U0_002.mp3
│   │   └── ...
│   └── no-emotion
│       ├── 5NsJZMbU9U0_001
│       │   ├── 5NsJZMbU9U0_001.json
│       │   └── 5NsJZMbU9U0_001.mp3
│       └── ...
└ ...
```

---

## How to run

```console
$ make install # create python environment and install "requirements.txt"
$ make run # creates segments with detected detection -> results/<video_id>.json
$ make dataset # creates dataset from results/
```

---

## Pipeline

The processing follows these main steps:

### 1. Audio Download
- Uses [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) to fetch the **best-quality audio** from YouTube.
- Files are cached in `.cache_pipeline/<video_id>/`.

### 2. Audio Preprocessing
- Splits long recordings into smaller chunks.
- Converts to **mono 16 kHz** using `torchaudio`.
- Extracts **vocals** from the tracks using [`demucs`](https://github.com/adefossez/demucs).
- Files are cached in `.cache_pipeline/<video_id>/`.

### 3. Speaker Diarization
- Runs **NVIDIA NeMo**’s `ClusteringDiarizer` to detect speakers.
- Produces time-aligned segments with speaker labels.

### 4. Transcription
- Uses **Whisper (turbo)** model to transcribe Czech speech with word timestamps.

### 5. Segment Alignment
- Aligns transcribed words with diarized speaker segments.
- Merges and filters segments by duration and text length for higher quality.

### 6. Emotion & Sentiment Analysis
Each text segment is analyzed by multiple models:
- **Czech Roberta (emotion/sentiment classifier)**
- **Local LLM via Ollama (Gemma 3 12B or similar)**
- **Optional Wav2Vec** (English-trained — less accurate for Czech)

**Decision logic:**  
- If Roberta confidence > 0.8 → trust Roberta  
- Otherwise → fallback to LLM output  

Each segment gets combined emotion metadata like:
```json
{
  "emotion": "fear",
  "sentiment": "negative",
  "score": 0.93
}
