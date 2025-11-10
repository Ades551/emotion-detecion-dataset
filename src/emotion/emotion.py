from transformers import pipeline
from datasets import Dataset
from speechbrain.inference.interfaces import foreign_class
from pathlib import Path
import time
import json
from pydub import AudioSegment
import os
from typing import TypedDict, Literal
from tqdm import tqdm
import requests

import ollama
import subprocess

from models import Segment

# -------------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------------
class Emotion(TypedDict):
    emotion: str
    sentiment: str
    score: float

PROMPT_TEMPLATE_SENTIMENT = """
Jsi expert na porozumění emocím a sentimentu v češtině.
Analyzuj následující český text, který mohl být automaticky přepsán z audia, takže může obsahovat překlepy, chybějící slova nebo špatně rozpoznaná slova.
Přesto se pokus pochopit původní význam a kontext.

Vrať JSON se dvěma poli:
- "sentiment": jedna z možností ["positive", "neutral", "negative"]
- "emotion": jedna z možností ["joy", "sadness", "anger", "fear", "surprise", "disgust", "none"]

Pokud je text sarkastický, humorný nebo nejasný kvůli chybám v přepisu, odhadni nejpravděpodobnější emoci a sentiment podle kontextu.
Text: "{text}"
Vrať pouze platný JSON, žádný další text.
"""

# -------------------------------------------------------------------------
# Emotion Analyzer Class
# -------------------------------------------------------------------------
class EmotionAnalyzer:
    """Unified interface for emotion & sentiment detection using multiple backends."""

    def __init__(
        self,
        backend: Literal["roberta", "wav2vec", "llm"] = "roberta",
        device: int = 0,
    ):
        self.backend = backend
        self.device = device

        # Preload model if applicable
        if backend == "roberta":
            self.model = pipeline("text-classification", model="visegradmedia-emotion/Emotion_RoBERTa_czech6", device=device)
        elif backend == "wav2vec":
            self.model = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier"
            )
        else:
            self.model = None

        # Label maps
        self.LABEL_TO_SENTIMENT = {
            "LABEL_0": "negative",
            "LABEL_1": "negative",
            "LABEL_2": "negative",
            "LABEL_3": "negative",
            "LABEL_4": "positive",
            "LABEL_5": "neutral",
        }

        self.LABEL_TO_EMOTION = {
            "LABEL_0": "anger",
            "LABEL_1": "fear",
            "LABEL_2": "disgust",
            "LABEL_3": "sadness",
            "LABEL_4": "joy",
            "LABEL_5": "none",
        }

        self.INDEX_TO_SENTIMENT = {
            0: "neutral",
            1: "negative",
            2: "positive",
            3: "negative",
        }

        self.INDEX_TO_EMOTION = {
            0: "none",
            1: "anger",
            2: "joy",
            3: "sadness",
        }

    # -------------------------------------------------------------------------
    # 1. Czech RoBERTa Text Model
    # -------------------------------------------------------------------------
    def analyze_roberta(self, texts: list[str]) -> list[Emotion]:
        dataset = Dataset.from_dict({"text": texts})
        preds = self.model(list(dataset["text"]), batch_size=64, truncation=True)

        return [
            Emotion(
                emotion=self.LABEL_TO_EMOTION[e["label"]],
                sentiment=self.LABEL_TO_SENTIMENT[e["label"]],
                score=float(e["score"]),
            )
            for e in preds
        ]
    

    # -------------------------------------------------------------------------
    # 2. Wav2Vec2 Audio Model
    # -------------------------------------------------------------------------
    def analyze_wav2vec(self, mono_16k_audio: Path, segments: list[Segment]) -> list[Emotion]:
        """Infer emotions directly from audio waveform."""
        audio = AudioSegment.from_file(mono_16k_audio)
        results = []

        for seg in tqdm(segments, desc="Analyzing with Wav2Vec2"):
            start_ms, end_ms = int(seg.start * 1000), int(seg.end * 1000)
            clip = audio[start_ms:end_ms]
            clip.export("./clip.wav", format="wav")

            out_prob, score, index, _ = self.model.classify_file("./clip.wav")
            results.append(
                Emotion(
                    emotion=self.INDEX_TO_EMOTION[index.item()],
                    sentiment=self.INDEX_TO_SENTIMENT[index.item()],
                    score=score.item(),
                )
            )

        os.unlink("./clip.wav")
        return results
    
    # -------------------------------------------------------------------------
    # 3. Local LLM via Ollama
    # -------------------------------------------------------------------------
    def analyze_llm(self, texts: list[str]) -> list[Emotion]:
        process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)

        results = []
        for t in tqdm(texts, desc="Analyzing with Ollama LLM"):
            response = ollama.chat(
                model="gemma3:12b",
                messages=[{"role": "user", "content": PROMPT_TEMPLATE_SENTIMENT.format(text=t)}],
                stream=False,
                format="json"
            )

            data = json.loads(response.message.content)

            results.append(
                Emotion(
                    emotion=data.get("emotion", "none"),
                    sentiment=data.get("sentiment", "neutral"),
                    score=1.0,
                )
            )

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:12b",
                "keep_alive": 0
            }
        )
        response.raise_for_status()

        subprocess.run(["ollama", "stop", "gemma3:12b"], capture_output=True, check=False)
        process.kill()
        time.sleep(5)
        return results

    # -------------------------------------------------------------------------
    # Unified Interface
    # -------------------------------------------------------------------------
    def analyze(self, segments: list[Segment], audio_file: Path | None = None) -> list[Emotion]:
        texts = [s.text for s in segments]

        if self.backend == "roberta":
            return self.analyze_roberta(texts)
        elif self.backend == "wav2vec":
            if audio_file is None:
                raise ValueError("audio_file must be provided for wav2vec backend.")
            return self.analyze_wav2vec(audio_file, segments)
        elif self.backend == "llm":
            return self.analyze_llm(texts)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")