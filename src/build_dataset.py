import json
import shutil
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import sys


def build_dataset(data_json_path: Path, output_dir: Path):
    """
    Build dataset folders with labeled audio clips from annotated segment metadata.
    
    Args:
        data_json_path: Path to the JSON file with structure shown above.
        output_dir: Root folder for the generated dataset.
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    yt_id = Path(data["recording"]).stem  # e.g., "AHUceoo-L6k"
    audio = AudioSegment.from_file(data["recording"])

    url = data["url"]

    # Prepare target root
    dataset_root = output_dir / yt_id
    shutil.rmtree(dataset_root, ignore_errors=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building dataset for {yt_id} ...")

    for i, seg in enumerate(tqdm(data["segments"], desc="Processing segments"), 1):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = audio[start_ms:end_ms]

        roberta_score = seg["emotion"]["czech_roberta"].get("score", 0.0)

        if roberta_score > 0.8:
            final_emotion = seg["emotion"]["czech_roberta"]["emotion"]
            final_sentiment = seg["emotion"]["czech_roberta"]["sentiment"]
            source_model = "czech_roberta"
        else:
            final_emotion = seg["emotion"]["llm"]["emotion"]
            final_sentiment = seg["emotion"]["llm"]["sentiment"]
            source_model = "llm"

        # --- Define folder paths ---
        label_folder = "no-emotion" if final_emotion == "none" else "emotion"
        clip_dir = dataset_root / label_folder / f"{yt_id}_{i:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        # --- Save clip ---
        clip_path = clip_dir / f"{yt_id}_{i:03d}.mp3"
        clip.export(clip_path, format="mp3")

        # --- Save metadata ---
        metadata = {
            "id": yt_id,
            "url": f"{url}&t={int(seg['start'])}s",
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["duration"],
            "text": seg["text"],
            "emotion_llm": seg["emotion"]["llm"],
            "emotion_roberta": seg["emotion"]["czech_roberta"],
            "final_emotion": final_emotion,
            "final_sentiment": final_sentiment,
            "source_model": source_model,
        }

        with open(clip_dir / f"{yt_id}_{i:03}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Dataset created at: {dataset_root}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_dataset.py <data_json_path>")
        sys.exit(1)

    data_json_path = Path(sys.argv[1])
    output_dir = Path("dataset")

    build_dataset(data_json_path, output_dir)