# Cell 2: Configuration, Demo File Download, Model Definitions, and MLP Path Discovery
# This demo is not speed optimized and therefore pretty slow.
# Atm it is loading every MLP model at a time,
# because the RAM and the VRAM of the GPU here in Google Colab are not big enough to keep all.

import os
import sys
import json
import gc
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import shutil
import requests # For downloading demo files

import torch
import torch.nn as nn
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download, hf_hub_download

# --- User-Modifiable Configuration ---
# For MLP model execution
USE_CPU_OFFLOADING_FOR_MLPS = True
USE_HALF_PRECISION_FOR_MLPS = True
USE_TORCH_COMPILE_FOR_MLPS = True

WHISPER_MODEL_ID = "mkrausio/EmoWhisper-AnS-Small-v0.1"
HF_MLP_REPO_ID = "laion/Empathic-Insight-Voice-Small"
LOCAL_MLP_MODELS_DOWNLOAD_DIR = Path("empathic_insight_voice_small_models_downloaded")

# --- Paths for Batch Processing ---
# Create these folders in your Colab environment if they don't exist
# !mkdir -p /content/batch_audio_input /content/batch_annotations_output_html /content/batch_annotations_output_json
BATCH_INPUT_AUDIO_FOLDER = Path("content/batch_audio_input")
BATCH_OUTPUT_HTML_REPORT_FILE = Path("content/batch_annotations_output_html/batch_emotion_report.html")
BATCH_OUTPUT_JSON_FOLDER = Path("content/batch_annotations_output_json")

DEMO_AUDIO_FILES_TO_DOWNLOAD = {
    "1.mp3": "https://huggingface.co/laion/Empathic-Insight-Voice-Small/resolve/main/1.mp3",
    "2.mp3": "https://huggingface.co/laion/Empathic-Insight-Voice-Small/resolve/main/2.mp3",
    "3.mp3": "https://huggingface.co/laion/Empathic-Insight-Voice-Small/resolve/main/3.mp3",
    "4.mp3": "https://huggingface.co/laion/Empathic-Insight-Voice-Small/resolve/main/4.mp3",
}

# --- Core Model & Audio Configuration ---
SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30.0
WHISPER_SEQ_LEN: int = 1500
WHISPER_EMBED_DIM: int = 768
PROJECTION_DIM_FOR_FULL_EMBED: int = 64
MLP_HIDDEN_DIMS: List[int] = [64, 32, 16]
MLP_DROPOUTS: List[float] = [0.0, 0.1, 0.1, 0.1]

SUPPORTED_AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac']

TARGET_EMOTION_KEYS_FOR_REPORT: List[str] = [
    "Amusement", "Elation", "Pleasure/Ecstasy", "Contentment", "Thankfulness/Gratitude",
    "Affection", "Infatuation", "Hope/Enthusiasm/Optimism", "Triumph", "Pride",
    "Interest", "Awe", "Astonishment/Surprise", "Concentration", "Contemplation",
    "Relief", "Longing", "Teasing", "Impatience and Irritability",
    "Sexual Lust", "Doubt", "Fear", "Distress", "Confusion", "Embarrassment", "Shame",
    "Disappointment", "Sadness", "Bitterness", "Contempt", "Disgust", "Anger",
    "Malevolence/Malice", "Sourness", "Pain", "Helplessness", "Fatigue/Exhaustion",
    "Emotional Numbness", "Intoxication/Altered States of Consciousness", "Jealousy / Envy"
]
assert len(TARGET_EMOTION_KEYS_FOR_REPORT) == 40

FILENAME_PART_TO_TARGET_KEY_MAP: Dict[str, str] = {
    "Affection": "Affection", "Age": "Age", "Amusement": "Amusement", "Anger": "Anger",
    "Arousal": "Arousal", "Astonishment_Surprise": "Astonishment/Surprise",
    "Authenticity": "Authenticity", "Awe": "Awe", "Background_Noise": "Background_Noise",
    "Bitterness": "Bitterness", "Concentration": "Concentration",
    "Confident_vs._Hesitant": "Confident_vs._Hesitant", "Confusion": "Confusion",
    "Contemplation": "Contemplation", "Contempt": "Contempt", "Contentment": "Contentment",
    "Disappointment": "Disappointment", "Disgust": "Disgust", "Distress": "Distress",
    "Doubt": "Doubt", "Elation": "Elation", "Embarrassment": "Embarrassment",
    "Emotional_Numbness": "Emotional Numbness", "Fatigue_Exhaustion": "Fatigue/Exhaustion",
    "Fear": "Fear", "Gender": "Gender", "Helplessness": "Helplessness",
    "High-Pitched_vs._Low-Pitched": "High-Pitched_vs._Low-Pitched",
    "Hope_Enthusiasm_Optimism": "Hope/Enthusiasm/Optimism",
    "Impatience_and_Irritability": "Impatience and Irritability",
    "Infatuation": "Infatuation", "Interest": "Interest",
    "Intoxication_Altered_States_of_Consciousness": "Intoxication/Altered States of Consciousness",
    "Jealousy_&_Envy": "Jealousy / Envy", "Longing": "Longing",
    "Malevolence_Malice": "Malevolence/Malice",
    "Monotone_vs._Expressive": "Monotone_vs._Expressive", "Pain": "Pain",
    "Pleasure_Ecstasy": "Pleasure/Ecstasy", "Pride": "Pride",
    "Recording_Quality": "Recording_Quality", "Relief": "Relief", "Sadness": "Sadness",
    "Serious_vs._Humorous": "Serious_vs._Humorous", "Sexual_Lust": "Sexual Lust",
    "Shame": "Shame", "Soft_vs._Harsh": "Soft_vs._Harsh", "Sourness": "Sourness",
    "Submissive_vs._Dominant": "Submissive_vs._Dominant", "Teasing": "Teasing",
    "Thankfulness_Gratitude": "Thankfulness/Gratitude", "Triumph": "Triumph",
    "Valence": "Valence",
    "Vulnerable_vs._Emotionally_Detached": "Vulnerable_vs._Emotionally_Detached",
    "Warm_vs._Cold": "Warm_vs._Cold"
}

# --- MLP Model Definition ---
class FullEmbeddingMLP(nn.Module): # (Same as before)
    def __init__(self,
                 seq_len: int,
                 embed_dim: int,
                 projection_dim: int,
                 mlp_hidden_dims: List[int],
                 mlp_dropout_rates: List[float]):
        super().__init__()
        if len(mlp_dropout_rates) != len(mlp_hidden_dims) + 1:
            raise ValueError(f"Dropout rates length error. Expected {len(mlp_hidden_dims) + 1}, got {len(mlp_dropout_rates)}")
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(seq_len * embed_dim, projection_dim)
        layers = [nn.ReLU(), nn.Dropout(mlp_dropout_rates[0])]
        current_dim = projection_dim
        for i, h_dim in enumerate(mlp_hidden_dims):
            layers.extend([
                nn.Linear(current_dim, h_dim), nn.ReLU(), nn.Dropout(mlp_dropout_rates[i+1])
            ])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[1] == 1: x = x.squeeze(1)
        return self.mlp(self.proj(self.flatten(x)))

# --- Helper Functions (Adapted) ---
def download_hf_mlp_checkpoints(repo_id: str, local_dir: Path, force_redownload: bool = False) -> Path:
    if local_dir.exists() and force_redownload:
        logging.info(f"Force redownload: Removing existing directory {local_dir}")
        shutil.rmtree(local_dir)
    if not local_dir.exists() or not any(local_dir.glob("*.pth")):
        logging.info(f"Downloading MLP checkpoints from {repo_id} to {local_dir}...")
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, allow_patterns=["*.pth"], repo_type="model")
            logging.info(f"MLP checkpoints downloaded to {local_dir}")
        except Exception as e:
            logging.error(f"Failed to download MLP checkpoints from {repo_id}: {e}", exc_info=True)
            raise RuntimeError(f"Could not download MLP checkpoints from {repo_id}.")
    else:
        logging.info(f"MLP checkpoints found in local cache: {local_dir}")
    return local_dir

def download_demo_audio_files(target_dir: Path, files_to_download: Dict[str, str]):
    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading demo audio files to {target_dir}...")
    for filename, url in files_to_download.items():
        filepath = target_dir / filename
        if filepath.exists():
            logging.info(f"Demo file {filename} already exists. Skipping download.")
            continue
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Downloaded {filename} to {filepath}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading {filename} from {url}: {e}")
        except IOError as e:
            logging.error(f"Error writing {filename} to {filepath}: {e}")

def get_mlp_model_paths_map(mlp_checkpoints_dir: Path, filename_map: Dict[str, str]) -> Dict[str, Path]:
    all_mapped_model_paths: Dict[str, Path] = {}
    if not mlp_checkpoints_dir.is_dir():
        logging.error(f"MLP checkpoints directory not found: {mlp_checkpoints_dir.resolve()}")
        return {}
    logging.info(f"Mapping MLP model files from {mlp_checkpoints_dir.resolve()}...")
    for pth_file in mlp_checkpoints_dir.glob("model_*_best.pth"):
        try:
            filename_part = pth_file.name.split("model_")[1].split("_best.pth")[0]
            if filename_part in filename_map:
                target_key = filename_map[filename_part]
                if target_key in all_mapped_model_paths:
                    logging.warning(f"Duplicate mapping for target key '{target_key}'. Overwriting.")
                all_mapped_model_paths[target_key] = pth_file
            # else:
                # logging.debug(f"Filename part '{filename_part}' not in map. Skipping {pth_file.name}")
        except IndexError:
            logging.warning(f"Could not parse filename part from {pth_file.name}. Skipping.")
    logging.info(f"Found {len(all_mapped_model_paths)} MLP model paths based on map.")
    return all_mapped_model_paths

def load_whisper_model(model_id: str, device: torch.device) -> Tuple[Optional[WhisperForConditionalGeneration], Optional[WhisperProcessor]]:
    logging.info(f"Loading Whisper model '{model_id}' to {device}...")
    try:
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        model.eval()
        logging.info(f"Whisper model '{model_id}' loaded to {device}.")
        return model, processor
    except Exception as e:
        logging.error(f"Error loading Whisper model '{model_id}': {e}", exc_info=True)
        return None, None

# --- Function to load a SINGLE MLP model ---
def load_single_mlp_model(
    model_path: Path,
    target_key: str, # For logging
    mlp_device: torch.device,
    use_half_cfg: bool,
    use_compile_cfg: bool
) -> Optional[nn.Module]:
    target_dtype = torch.float16 if use_half_cfg and mlp_device.type == 'cuda' else torch.float32
    compile_mode = "reduce-overhead"
    logging.debug(f"Loading MLP for '{target_key}' from {model_path} to {mlp_device} (Half: {use_half_cfg}, Compile: {use_compile_cfg})")
    try:
        model_instance = FullEmbeddingMLP(
            seq_len=WHISPER_SEQ_LEN, embed_dim=WHISPER_EMBED_DIM,
            projection_dim=PROJECTION_DIM_FOR_FULL_EMBED,
            mlp_hidden_dims=MLP_HIDDEN_DIMS,
            mlp_dropout_rates=MLP_DROPOUTS
        )
        state_dict_content = torch.load(model_path, map_location='cpu')
        actual_state_dict = state_dict_content # Assuming raw state_dict from HF

        needs_stripping = any(k.startswith("_orig_mod.") for k in actual_state_dict.keys())
        if needs_stripping:
            stripped_state_dict = {
                k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                for k, v in actual_state_dict.items()
            }
            actual_state_dict = stripped_state_dict

        model_instance.load_state_dict(actual_state_dict)
        model_instance.eval()

        if mlp_device.type == 'cuda' and use_half_cfg:
            model_instance = model_instance.to(dtype=target_dtype)
        model_instance = model_instance.to(mlp_device)

        if use_compile_cfg and hasattr(torch, 'compile') and mlp_device.type == 'cuda' and torch.__version__ >= "2.0.0":
            try:
                model_instance = torch.compile(model_instance, mode=compile_mode)
            except Exception as e_compile:
                logging.warning(f"torch.compile failed for MLP '{target_key}': {e_compile}. Using uncompiled.")

        logging.debug(f"Successfully loaded MLP for '{target_key}'.")
        return model_instance
    except Exception as e:
        logging.error(f"Failed to load/prepare MLP for '{target_key}' from {model_path}: {e}", exc_info=True)
        return None

# --- Determine Devices ---
_whisper_device_type = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_DEVICE = torch.device(_whisper_device_type)
_mlp_device_type = "cpu" if USE_CPU_OFFLOADING_FOR_MLPS else _whisper_device_type
MLP_DEVICE = torch.device(_mlp_device_type)

logging.info(f"Whisper will run on: {WHISPER_DEVICE}")
logging.info(f"MLPs will run on: {MLP_DEVICE}")

# --- Preparations ---
# 1. Create output directories
BATCH_INPUT_AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
BATCH_OUTPUT_HTML_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
BATCH_OUTPUT_JSON_FOLDER.mkdir(parents=True, exist_ok=True)

# 2. Download demo audio files into the BATCH_INPUT_AUDIO_FOLDER
#download_demo_audio_files(BATCH_INPUT_AUDIO_FOLDER, DEMO_AUDIO_FILES_TO_DOWNLOAD)

# 3. Download MLP checkpoints from Hugging Face
downloaded_mlp_checkpoints_dir = download_hf_mlp_checkpoints(HF_MLP_REPO_ID, LOCAL_MLP_MODELS_DOWNLOAD_DIR)

# 4. Load Whisper Model (this stays loaded)
whisper_model_global, whisper_processor_global = load_whisper_model(WHISPER_MODEL_ID, WHISPER_DEVICE)
if not whisper_model_global or not whisper_processor_global:
    raise RuntimeError("Failed to load Whisper model. Cannot proceed.")

# 5. Get paths and map for all MLP models (these will be loaded one by one later)
# This dictionary {target_key: model_path_object} is crucial for Cells 3 & 4.
all_mlp_model_paths_dict: Dict[str, Path] = get_mlp_model_paths_map(
    downloaded_mlp_checkpoints_dir,
    FILENAME_PART_TO_TARGET_KEY_MAP
)
if not all_mlp_model_paths_dict:
    raise RuntimeError("No MLP model paths could be mapped or found. Cannot proceed.")

logging.info(f"--- Cell 2 Setup Complete. Whisper model loaded. {len(all_mlp_model_paths_dict)} MLP model paths identified. ---")
logging.info(f"Demo audio files are in: {BATCH_INPUT_AUDIO_FOLDER.resolve()}")
logging.info(f"HTML report will be saved to: {BATCH_OUTPUT_HTML_REPORT_FILE.resolve()}")
logging.info(f"JSON annotations will be saved to: {BATCH_OUTPUT_JSON_FOLDER.resolve()}")