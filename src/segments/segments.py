from pathlib import Path
import shutil
import json
import logging

from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
from models import Segment
import whisper
import torch
import random
import numpy as np
import gc

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SegmentDetection:
    def detect_segments(self, vocal_audio: Path, mono_16k_audio: Path, max_gap_treshold: float = 1.0) -> list[Segment]:
        segments = self.diarize_recording(mono_16k_audio)
        words = self.transcribe(vocal_audio)
        segments = self.align_words(words, segments)
        segments = self.merge_segments(segments=segments, max_gap_treshold=max_gap_treshold)
        segments = self.filter_segments(segments=segments)

        return segments

    def filter_segments(
            self, 
            segments: list[Segment],
            min_duration_sec: float = 8.0,
            max_duration_sec: float = 30.0,
            min_text_lenght: int = 80,
        ) -> list[Segment]:

        best_segments = []
        for segment in segments:
            if min_duration_sec <= segment.duration <= max_duration_sec and len(segment.text) >= min_text_lenght:
                best_segments.append(segment)

        return best_segments

    def merge_segments(self, segments: list[Segment], max_gap_treshold: float) -> list[Segment]:
        merged_segments = [segments[0]]

        for curr_seg in segments[1:]:
            last_merged = merged_segments[-1]

            gap = curr_seg.start - last_merged.end

            # 1. Are they the same speaker? (e.g., speaker_0 followed by speaker_0)
            is_same_speaker = (last_merged.speaker == curr_seg.speaker)
            
            # 2. Is one of them UNKNOWN? (The case we want to fix)
            one_is_unknown = (last_merged.speaker == 'UNKNOWN' or 
                                curr_seg.speaker == 'UNKNOWN')
            
            # 3. Are they close enough in time?
            is_close_enough = (gap <= max_gap_treshold)

            if is_close_enough and (is_same_speaker or one_is_unknown):
                last_merged.text += " " + curr_seg.text
                last_merged.end = curr_seg.end

                if last_merged.speaker == 'UNKNOWN' and curr_seg.speaker != 'UNKNOWN':
                    last_merged.speaker = curr_seg.speaker
            else:
                merged_segments.append(
                    Segment(
                        speaker=curr_seg.speaker,
                        text=curr_seg.text.strip(),
                        start=curr_seg.start,
                        end=curr_seg.end
                    )
                )

        return merged_segments

    def align_words(self, words: list[dict[str, float | str]], segments: list[Segment]) -> list[Segment]:
        """Assign words directly to segments based on time overlap."""
        sentence_boundaries = {".", "?", "!"}
        sentences = []
        current_sentence = []

        # --- 1. Group words into sentences ---
        for word in words:
            current_sentence.append(word)
            if any(word["word"].strip().endswith(p) for p in sentence_boundaries):
                sentences.append(current_sentence)
                current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)

        # --- 2. Assign sentences to the segments ---
        for seg in segments:
            for sentence in sentences:
                start = sentence[0]["start"]
                end = sentence[-1]["end"]

                if seg.start <= ((start + end) / 2) <= seg.end:
                    seg.text += ' '.join([w["word"] for w in sentence])
                    seg.text += " "

            seg.text = seg.text.strip()

        return segments

    def transcribe(self, vocal_audio: Path) -> list[dict[str, float | str]]:
        model = whisper.load_model("turbo")
        
        result = model.transcribe(
            str(vocal_audio.absolute()),
            language="cs", 
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
            word_timestamps=True
        )

        all_words: list[dict[str, float | str]] = []

        for seg in result["segments"]:
            if "words" not in seg:
                continue

            for word_info in seg["words"]:
                word_info["word"] = word_info["word"].strip()
                all_words.append(word_info)

        torch.cuda.empty_cache()
        gc.collect()

        return all_words


    def diarize_recording(self, mono_16k_audio: Path) -> list[Segment]:
        temp_dir = Path(".temp")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------------
        # 1. Build NeMo manifest file
        # -------------------------------------------------------------------------
        manifest_path = temp_dir / "input_manifest.json"
        meta = {
            "audio_filepath": str(mono_16k_audio.absolute()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,   # or set a number if you want to enforce it
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open(manifest_path, "w") as f:
            json.dump(meta, f)
            f.write("\n")

        # -------------------------------------------------------------------------
        # 2. Load diarization config (diar_infer_general.yaml)
        # -------------------------------------------------------------------------
        config_path = Path(__file__).parent / "diar_infer_general.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing diarization config file: {config_path}. "
                "Download one from NeMo examples or your repo."
            )

        cfg = OmegaConf.load(config_path)
        # Update paths dynamically
        cfg.diarizer.manifest_filepath = str(manifest_path)
        cfg.diarizer.out_dir = str(temp_dir)

        # Optional: disable oracle VAD or speaker count if present
        if "oracle_vad" in cfg.diarizer:
            cfg.diarizer.oracle_vad = False
        if "oracle_num_speakers" in cfg.diarizer:
            cfg.diarizer.oracle_num_speakers = False

        # -------------------------------------------------------------------------
        # 3. Run diarization
        # -------------------------------------------------------------------------
        diarizer = ClusteringDiarizer(cfg=cfg)
        diarizer.diarize()

        # -------------------------------------------------------------------------
        # 4. Get RTTM path
        # -------------------------------------------------------------------------
        rttm_dir = temp_dir / "pred_rttms"
        rttm_files = list(rttm_dir.glob("*.rttm"))
        if not rttm_files:
            raise FileNotFoundError(f"No RTTM files generated in {rttm_dir}")
        
        # -------------------------------------------------------------------------
        # 5. Create Segments
        # -------------------------------------------------------------------------
        speaker_segments: list[Segment] = []

        for line in rttm_files[0].read_text().splitlines():
            data = line.strip().split(" ")
            start = float(data[5])
            speaker_segments.append(Segment(speaker=data[11], start=start, end=start + float(data[8])))
        
        shutil.rmtree(temp_dir, ignore_errors=True)

        return speaker_segments
