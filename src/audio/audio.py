import yt_dlp
from pathlib import Path
import tqdm
import json
from pydub import AudioSegment
import torch
import torchaudio
import demucs.separate
import shutil


class ProcessAudio:
    audio_path: Path
    audio_chunks_16k: list[Path]
    audio_vocals: list[Path]
    chunk_duration_ms: int

    def __init__(self, cahce_dir: Path):
        self.cache_dir = cahce_dir

    def initialize(self, url: str, chunk_duration_ms: int) -> None:
        """Initialize processing (download, split, preprocess, cache metadata)."""
        self.audio_path = self.download_youtube_audio(url)
        video_id = self.audio_path.stem
        video_dir = self.cache_dir / video_id
        metadata_path = video_dir / "metadata.json"
        self.chunk_duration_ms = chunk_duration_ms

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            # Check if split size matches
            if self.metadata.get("chunk_duration_ms") == chunk_duration_ms:
                print(f"Using cached metadata for {video_id}")
                self.audio_vocals = [Path(p) for p in self.metadata["chunks_vocal"]]
                self.audio_chunks_16k = [Path(p) for p in self.metadata["chunks_16k"]]
                self.chunk_duration_ms = self.metadata["chunk_duration_ms"]
                return
            else:
                print("Split size changed, reprocessing audio...")

        # Step 1: Split
        audio_chunks = self.split(chunk_duration_ms)

        # Step 2: Separate vocals
        self.audio_vocals = self.vocals(audio_chunks)

        # Step 3: Preprocess (mono, 16kHz)
        self.audio_chunks_16k = self.preprocess(self.audio_vocals)

        # Step 4: Save metadata
        self.metadata = {
            "url": url,
            "recording": str(self.audio_path),
            "chunk_duration_ms": chunk_duration_ms,
            "chunks_vocal": [str(p) for p in self.audio_vocals],
            "chunks_16k": [str(p) for p in self.audio_chunks_16k],
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"Metadata saved to {metadata_path}")

        shutil.rmtree(audio_chunks[0].parent, ignore_errors=True)

    def download_youtube_audio(self, url: str) -> Path:
        """Download YouTube audio as MP3 (if not already downloaded) and return the file path."""
        # Use video ID for a stable, filesystem-safe name
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get("id")

        (self.cache_dir / video_id).mkdir(exist_ok=True)

        audio_path = self.cache_dir / video_id / f"{video_id}.mp3"

        # If already downloaded, skip
        if audio_path.exists():
            print(f"Audio already downloaded: {audio_path}")
            return audio_path
        
        # Otherwise download
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(audio_path.with_suffix(".%(ext)s")),
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
            ],
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # yt_dlp may rename the file (due to ext replacement), so find it safely
        if not audio_path.exists():
            raise FileNotFoundError(f"Failed to find: {str(audio_path)}")

        return audio_path

    def split(self, chunk_duration_ms: int) -> list[Path]:
        """Split audio into fixed-length chunks (e.g., 10 minutes)."""
        print("Splitting audio into chunks...")
        audio = AudioSegment.from_file(self.audio_path)
        duration = len(audio)
        chunks = []
        output_dir = self.audio_path.parent / "chunks"
        output_dir.mkdir(exist_ok=True)

        for i, start in enumerate(tqdm.tqdm(range(0, duration, chunk_duration_ms), desc="Splitting")):
            end = min(start + chunk_duration_ms, duration)
            chunk = audio[start:end]
            chunk_path = output_dir / f"{self.audio_path.stem}_part{i+1:02d}.mp3"
            chunk.export(chunk_path, format="mp3")
            chunks.append(chunk_path)

        chunks.sort(key=lambda p: p.name)
        return chunks
        

    def preprocess(self, audio_chunks: list[Path]) -> list[Path]:
        """Convert audio chunks to mono + 16kHz WAV files using torchaudio (fast + consistent)."""
        print("Preprocessing chunks (mono + 16kHz using torchaudio)...")

        processed_dir = self.audio_path.parent / "chunks_16k"
        processed_dir.mkdir(exist_ok=True, parents=True)
        processed = []

        for audio_file in tqdm.tqdm(audio_chunks, desc="Preprocessing"):
            try:
                # Load waveform
                waveform, sr = torchaudio.load(audio_file)

                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Resample if needed
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    waveform = resampler(waveform)

                # Save to disk
                processed_path = processed_dir / f"{audio_file.stem}_16k.wav"
                torchaudio.save(processed_path, waveform, 16000)
                processed.append(processed_path)

            except Exception as e:
                print(f"Failed to preprocess {audio_file}: {e}")

        processed.sort(key=lambda p: p.name)
        return processed
    
    def vocals(self, audio_chunks: list[Path]) -> list[Path]:
        shutil.rmtree(".demucs/", ignore_errors=True)

        processed_dir = self.audio_path.parent / "vocals"
        processed_dir.mkdir(exist_ok=True, parents=True)
        processed = []

        for audio_file in audio_chunks:
            demucs.separate.main(["-n", "htdemucs", "--two-stems", "vocals", str(audio_file.absolute()), "-o", ".demucs", "--device", "cuda"])
            
            result = Path(f".demucs/htdemucs/{audio_file.stem}/vocals.wav")
            shutil.copyfile(result, processed_dir / f"{audio_file.stem}.wav")
            processed.append(processed_dir / f"{audio_file.stem}.wav")

        shutil.rmtree(".demucs/", ignore_errors=True)

        processed.sort(key=lambda p: p.name)
        return processed
