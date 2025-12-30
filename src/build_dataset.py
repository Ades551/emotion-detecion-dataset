import json
import shutil
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import sys
from LAION.predict import predict as predict_emotion, mlps

def merge_models_attributes(json_models_attrs):
    """
    Helper function to merge emotion scores from multiple models, 
    preparing data for the emotion prediction model.
    """
    merged_attrs = {}
    # Convert LLM 'none'/'emotion' label to 0.0/1.0 score
    merged_attrs["llm"] = 1.0 if json_models_attrs["llm"].get("emotion", "none") != "none" else 0.0
    # Extract scores for other models, setting to 0.0 if 'none'
    merged_attrs["czech_roberta"] = json_models_attrs["czech_roberta"].get("score", 0.0) if json_models_attrs["czech_roberta"].get("emotion", "none") != "none" else 0.0
    merged_attrs["wav2vec"] = json_models_attrs["wav2vec"].get("score", 0.0) if json_models_attrs["wav2vec"].get("emotion", "none") != "none" else 0.0 
    # Merge with LAION attributes
    return merged_attrs | json_models_attrs["laion"]

def build_dataset(data_json_path: Path, output_dir: Path):
    """
    Build dataset folders with labeled audio clips using a flat structure:
    output_dir/emotion/ and output_dir/no-emotion/.
    
    Args:
        data_json_path: Path to the JSON file with segment metadata.
        output_dir: Root folder for the generated dataset (e.g., "dataset").
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use the filename of the recording as the YouTube ID
    yt_id = Path(data["recording"]).stem 
    audio = AudioSegment.from_file(data["recording"])

    # Define the final target directories
    emotion_dir = output_dir / "emotion"
    no_emotion_dir = output_dir / "no-emotion"
    
    # Ensure the target directories exist
    emotion_dir.mkdir(parents=True, exist_ok=True)
    no_emotion_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Building dataset for {yt_id} into {output_dir.name}/(emotion|no-emotion) ...")

    total_duration_sec = 0.0

    for seg in tqdm(data["segments"], desc="Processing segments"):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        
        total_duration_sec += (seg["end"] - seg["start"])

        clip = audio[start_ms:end_ms]

        # 1. Predict final emotion label
        emotion_pred = predict_emotion(merge_models_attributes(seg["emotion"]))
        final_emotion = "emotion" if emotion_pred else "none"
            
        # 2. Determine target directory based on the final emotion
        clip_dir = emotion_dir if final_emotion == "emotion" else no_emotion_dir
        
        # --- START OF FILENAME CHANGE ---
        # 3. Use millisecond timestamps for filename precision
        # Example values: start_ms = 71775, end_ms = 88775
        
        # 4. Define the new filename format
        # Example: frame_AHUceoo-L6k_71775ms_88775ms.mp3
        clip_filename = f"{yt_id}_{start_ms:08d}_{end_ms:08d}.mp3"
        # --- END OF FILENAME CHANGE ---
        
        clip_path = clip_dir / clip_filename

        # 5. Save clip
        clip.export(clip_path, format="mp3")

        # Note: The step to save metadata is intentionally skipped as requested.

    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = int(total_duration_sec % 60)
    print(f"[INFO] Total dataset length: {hours}h {minutes}m {seconds}s")

    print(f"[INFO] Dataset created. Clips are organized in {output_dir.name}/emotion and {output_dir.name}/no-emotion.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_dataset.py <data_json_path>")
        sys.exit(1)

    data_json_path = Path(sys.argv[1])
    output_dir = Path("dataset")

    build_dataset(data_json_path, output_dir)