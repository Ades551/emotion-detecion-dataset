# Cell 1: Setup and Dependencies
#pip install transformers torch torchaudio torchvision librosa huggingface_hub numpy pydub ipython --quiet
print("Dependencies installed.")

import os
import sys
import json
import gc
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set
import base64
import io
import shutil # For cleaning up downloaded models if needed

import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display # For plotting waveform in HTML
import matplotlib.pyplot as plt # For plotting waveform

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download, hf_hub_download
from IPython.display import HTML, display, Audio as IPythonAudio

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError, CouldntEncodeError
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("WARNING: pydub library not found. Audio player in HTML report (Cell 3) will not have playable audio. Install with: !pip install pydub")
except Exception as e:
    PYDUB_AVAILABLE = False
    print(f"WARNING: Error initializing pydub (likely ffmpeg/avconv issue): {e}. Audio player in HTML report (Cell 3) will not have playable audio.")

# Setup basic logging
def setup_notebook_logging(log_level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)-7s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
setup_notebook_logging()

# --- Global Configuration Toggles (User can modify these in Cell 2) ---
# These will be properly defined and used in Cell 2.
# This is just a placeholder comment.