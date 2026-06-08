"""
audio_processor.py — Fixed version with chunked transcription for long audio
"""

import logging
import math
import os
import subprocess
import tempfile
import wave

import speech_recognition as sr
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".wav", ".mp3", ".mp4", ".webm", ".ogg", ".m4a", ".flac"}

# Google free STT works best with chunks under 50s
CHUNK_SECONDS = 25
OVERLAP_SECONDS = 1  # small overlap to avoid cutting mid-word at chunk boundary


# ─────────────────────────────────────────────
# 1. Convert to WAV  (16kHz mono s16)
# ─────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    wav_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        wav_path,
        "-loglevel", "error",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
        return wav_path

    logger.warning("ffmpeg conversion failed, using original file")
    return input_path


# ─────────────────────────────────────────────
# 2a. Split WAV into chunks
# ─────────────────────────────────────────────

def split_wav_into_chunks(wav_path: str, chunk_seconds: int = CHUNK_SECONDS) -> list[str]:
    """
    Split a WAV file into fixed-size chunks using ffmpeg.
    Returns list of temp file paths.
    """
    chunk_paths = []

    with wave.open(wav_path, "rb") as wf:
        total_frames = wf.getnframes()
        framerate = wf.getframerate()
        total_seconds = total_frames / framerate

    # No need to split short audio
    if total_seconds <= chunk_seconds:
        return [wav_path]

    num_chunks = math.ceil(total_seconds / chunk_seconds)
    logger.info(f"Splitting {total_seconds:.1f}s audio into {num_chunks} chunks of {chunk_seconds}s")

    for i in range(num_chunks):
        start = i * chunk_seconds
        # Slight overlap except on last chunk to avoid cutting words
        duration = chunk_seconds + (OVERLAP_SECONDS if i < num_chunks - 1 else 0)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        chunk_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-ss", str(start),
            "-t", str(duration),
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            chunk_path,
            "-loglevel", "error",
        ]
        proc = subprocess.run(cmd, capture_output=True)

        if proc.returncode == 0 and os.path.getsize(chunk_path) > 100:
            chunk_paths.append(chunk_path)
        else:
            logger.warning(f"Chunk {i} failed to generate, skipping")
            os.unlink(chunk_path)

    return chunk_paths


# ─────────────────────────────────────────────
# 2b. Transcribe a single WAV chunk
# ─────────────────────────────────────────────

def transcribe_chunk(wav_path: str, recognizer: sr.Recognizer) -> str:
    """
    Transcribe one WAV chunk. Returns empty string if speech not found in chunk
    (so a silent/unclear segment doesn't kill the whole transcription).
    """
    with sr.AudioFile(wav_path) as source:
        # Do NOT use adjust_for_ambient_noise — it consumes audio frames
        # and is unreliable on short chunks. Energy threshold handles this.
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data, language="ml-IN")
    except sr.UnknownValueError:
        logger.warning(f"No speech detected in chunk: {wav_path}")
        return ""  # Non-fatal — keep going with other chunks


# ─────────────────────────────────────────────
# 2. Transcribe full audio (handles long files)
# ─────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    """
    Transcribe a WAV file. Automatically chunks long audio.
    Raises sr.RequestError if Google API is unreachable on any chunk.
    Raises sr.UnknownValueError only if ALL chunks return empty.
    """
    recognizer = sr.Recognizer()
    # Fixed energy threshold — don't let dynamic threshold eat soft speech
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = False
    # Longer pause tolerance for natural Malayalam speech rhythm
    recognizer.pause_threshold = 1.2
    recognizer.phrase_threshold = 0.5
    recognizer.non_speaking_duration = 0.4

    chunk_paths = split_wav_into_chunks(wav_path, chunk_seconds=CHUNK_SECONDS)
    is_single_chunk = chunk_paths == [wav_path]  # wasn't split

    results = []
    try:
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
            text = transcribe_chunk(chunk_path, recognizer)
            if text:
                results.append(text)
    finally:
        # Clean up chunk temp files (but not the original wav_path)
        if not is_single_chunk:
            for p in chunk_paths:
                if p != wav_path and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    combined = " ".join(results).strip()

    if not combined:
        raise sr.UnknownValueError()  # All chunks were silent/unclear

    return combined


# ─────────────────────────────────────────────
# 3. Translate  (Malayalam → English)
# ─────────────────────────────────────────────

def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return ""
    return GoogleTranslator(source="ml", target="en").translate(text) or ""


# ─────────────────────────────────────────────
# 4. Pipeline orchestrator  (unchanged interface)
# ─────────────────────────────────────────────

def run_pipeline(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    translate: bool = True,
) -> dict:
    suffix = os.path.splitext(filename)[1].lower() or ".webm"
    if suffix not in SUPPORTED_SUFFIXES:
        return {
            "success": False, "malayalam": "", "english": "",
            "error": f"Unsupported format '{suffix}'. Supported: {', '.join(SUPPORTED_SUFFIXES)}",
        }

    tmp_path = wav_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        wav_path = convert_to_wav(tmp_path)

        try:
            malayalam_text = transcribe(wav_path)
        except sr.UnknownValueError:
            return {
                "success": False, "malayalam": "", "english": "",
                "error": "Speech not clear. Please speak closer to the microphone.",
            }
        except sr.RequestError as exc:
            return {
                "success": False, "malayalam": "", "english": "",
                "error": f"Google Speech API error: {exc}",
            }

        english_text = ""
        if translate:
            try:
                english_text = translate_to_english(malayalam_text)
            except Exception as exc:
                logger.error(f"Translation error: {exc}")
                return {
                    "success": True,
                    "malayalam": malayalam_text,
                    "english": "",
                    "error": f"Translation failed: {exc}",
                }

        return {
            "success": True,
            "malayalam": malayalam_text,
            "english": english_text,
            "error": None,
        }

    except Exception as exc:
        logger.exception("Audio processing error")
        return {
            "success": False, "malayalam": "", "english": "",
            "error": f"Processing error: {exc}",
        }

    finally:
        for path in (tmp_path, wav_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass