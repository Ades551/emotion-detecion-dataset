import sys
import gc
from models import Segment
from emotion import EmotionAnalyzer
from audio import ProcessAudio
from segments import SegmentDetection

from pathlib import Path
from dataclasses import asdict
import json
import torch
from time import sleep

torch.cuda.empty_cache()
gc.collect()

sleep(5)

CHUNK_DURATION_MS = 5 * 60 * 1000  # 5 minutes

def apply_time_offset(segments: list[Segment], offset_seconds: float) -> list[Segment]:
    """Shift segment timestamps by a given offset."""
    for s in segments:
        s.start += offset_seconds
        s.end += offset_seconds
    return segments


def segment_to_dict(s):
    d = asdict(s)

    # Drop unwanted fields if present
    for key in ("speaker", "label"):
        d.pop(key, None)
    
    # Always include duration
    d["duration"] = getattr(s, "duration", s.end - s.start)

    return d


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_audio.py <file-path-or-yt-url>")
        sys.exit(1)

    input_arg = sys.argv[1]
    input_path: Path

    # Step 1: Handle YouTube URL or local file
    if not input_arg.startswith("http"):
        print("Invalid input!")
        sys.exit(1)

    
    cache_dir = Path(".cache_pipeline/")
    cache_dir.mkdir(exist_ok=True)

    audio = ProcessAudio(cahce_dir=cache_dir)
    audio.initialize(input_arg, chunk_duration_ms=CHUNK_DURATION_MS)

    all_segments: list[Segment] = []
    s_det = SegmentDetection()

    analyzer_wav2vec = EmotionAnalyzer(backend="wav2vec")
    result_wav2vec = []

    analyzer_laion = EmotionAnalyzer(backend="laion")
    result_laion = []

    for i, (vocal_audio, mono_16k_audio) in enumerate(zip(audio.audio_vocals, audio.audio_chunks_16k)):
        offset_s = i * (audio.chunk_duration_ms / 1000.0)

        try:
            segments = s_det.detect_segments(vocal_audio=vocal_audio, mono_16k_audio=mono_16k_audio)
            result_wav2vec.extend(analyzer_wav2vec.analyze(segments, mono_16k_audio))
            result_laion.extend(analyzer_laion.analyze(segments, mono_16k_audio))
            
            segments = apply_time_offset(segments=segments, offset_seconds=offset_s)
            all_segments.extend(segments)

        except Exception as e:
            print(f"Skipping chunk {i+1} due to error: {e}")

    print(all_segments)

    analyzer = EmotionAnalyzer(backend="llm")
    result_llm = analyzer.analyze(all_segments)

    torch.cuda.empty_cache()
    gc.collect()

    analyzer = EmotionAnalyzer(backend="roberta")
    result_roberta = analyzer.analyze(all_segments)

    for i, segment in enumerate(all_segments):
        # {"czech_roberta": cz_roberta[idx]} | {"wav2vec": wav2vec[idx]} | {"llm": llm[idx]}
        segment.emotion |= {"llm": result_llm[i]} | {"czech_roberta": result_roberta[i]} | {"wav2vec": result_wav2vec[i]} | {"laion": result_laion[i]}

    output_dir = Path("results/")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{audio.audio_path.stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "url": input_arg,
                "recording": str(audio.audio_path.absolute()),
                "segments": [segment_to_dict(s) for s in all_segments]
            }, f, indent=2, ensure_ascii=False)

    print(f"Finished. Results saved to {output_path}")
