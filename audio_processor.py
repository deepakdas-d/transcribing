"""
audio_processor.py — Chunked transcription + remote Ollama medical extraction

Pipeline:
  1. convert_to_wav()          — ffmpeg: any format → 16kHz mono WAV
  2. split_wav_into_chunks()   — splits audio > CHUNK_SECONDS into overlapping slices
  3. transcribe()              — Google Speech Recognition (ml-IN), chunk-aware
  4. extract_medical_info()    — POST transcript → remote Colab Ollama → MedicalExtraction
  5. run_pipeline()            — orchestrator used by all endpoints
"""

import logging
import math
import os
import subprocess
import tempfile
import wave
import json
import requests

import speech_recognition as sr
from deep_translator import GoogleTranslator
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".wav", ".mp3", ".mp4", ".webm", ".ogg", ".m4a", ".flac"}

# Google free STT works best with chunks under 50s
CHUNK_SECONDS   = 25
OVERLAP_SECONDS = 1   # small overlap to avoid cutting mid-word at chunk boundary

# ── Remote Ollama (Google Colab via ngrok) ────────────────────────────────
COLAB_BASE_URL  = "https://repair-dolly-dollop.ngrok-free.dev"
OLLAMA_API_URL  = f"{COLAB_BASE_URL}/extract"   # Colab exposes /extract, not /ollama/api/generate
OLLAMA_MODEL    = "qwen3:8b"
OLLAMA_TIMEOUT  = 120   # seconds — Qwen on first call can be slow


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class MedicineItem(BaseModel):
    name: str = ""
    dose: str = ""
    frequency: str = ""
    duration: str = ""
    instructions: str = ""


class MedicalExtraction(BaseModel):
    """Structured medical info extracted from a transcript."""
    chief_complaints:          list[str]          = Field(default_factory=list)
    patient_reported_symptoms: list[str]          = Field(default_factory=list)
    symptoms:                  list[str]          = Field(default_factory=list)
    past_conditions_mentioned: list[str]          = Field(default_factory=list)
    conditions_mentioned:      list[str]          = Field(default_factory=list)
    medications_mentioned:     list[str]          = Field(default_factory=list)
    prescribed_medications:    list[MedicineItem] = Field(default_factory=list)
    body_parts_mentioned:      list[str]          = Field(default_factory=list)
    duration:                  list[str]          = Field(default_factory=list)
    severity:                  list[str]          = Field(default_factory=list)
    doctor_observations:       list[str]          = Field(default_factory=list)
    doctor_confirmed_diagnosis:list[str]          = Field(default_factory=list)
    advice:                    list[str]          = Field(default_factory=list)
    recommended_tests:         list[str]          = Field(default_factory=list)
    follow_up:                 list[str]          = Field(default_factory=list)
    risk_flags:                list[str]          = Field(default_factory=list)
    uncertain_items:           list[str]          = Field(default_factory=list)


# ─────────────────────────────────────────────
# 1. Convert to WAV  (16kHz mono s16)
# ─────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """
    Convert any browser audio format (webm/ogg/mp4) to 16kHz mono WAV via ffmpeg.
    Returns path to the WAV file, or original path if conversion fails.
    """
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
    Returns list of temp file paths (or [wav_path] if audio is short enough).
    """
    with wave.open(wav_path, "rb") as wf:
        total_frames  = wf.getnframes()
        framerate     = wf.getframerate()
        total_seconds = total_frames / framerate

    if total_seconds <= chunk_seconds:
        return [wav_path]

    num_chunks  = math.ceil(total_seconds / chunk_seconds)
    chunk_paths = []
    logger.info(f"Splitting {total_seconds:.1f}s audio into {num_chunks} chunks of {chunk_seconds}s")

    for i in range(num_chunks):
        start    = i * chunk_seconds
        duration = chunk_seconds + (OVERLAP_SECONDS if i < num_chunks - 1 else 0)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        chunk_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-ss", str(start),
            "-t",  str(duration),
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
            try:
                os.unlink(chunk_path)
            except OSError:
                pass

    return chunk_paths


# ─────────────────────────────────────────────
# 2b. Transcribe a single WAV chunk
# ─────────────────────────────────────────────

def transcribe_chunk(wav_path: str, recognizer: sr.Recognizer) -> str:
    """
    Transcribe one WAV chunk. Returns empty string if speech not found
    so a silent/unclear chunk doesn't abort the full transcription.
    """
    with sr.AudioFile(wav_path) as source:
        # Do NOT use adjust_for_ambient_noise — it consumes audio frames.
        # Fixed energy_threshold (set on the recognizer) handles noise.
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data, language="ml-IN")
    except sr.UnknownValueError:
        logger.warning(f"No speech detected in chunk: {wav_path}")
        return ""


# ─────────────────────────────────────────────
# 2. Transcribe full audio (chunk-aware)
# ─────────────────────────────────────────────

def transcribe(wav_path: str) -> str:
    """
    Transcribe a WAV file, automatically chunking long audio.

    Raises:
        sr.RequestError      — Google Speech API unreachable (any chunk)
        sr.UnknownValueError — ALL chunks returned empty text
    """
    recognizer = sr.Recognizer()
    recognizer.energy_threshold         = 300
    recognizer.dynamic_energy_threshold = False   # fixed threshold — don't swallow soft speech
    recognizer.pause_threshold          = 1.2     # tolerate natural Malayalam speech pauses
    recognizer.phrase_threshold         = 0.5
    recognizer.non_speaking_duration    = 0.4

    chunk_paths     = split_wav_into_chunks(wav_path, chunk_seconds=CHUNK_SECONDS)
    is_single_chunk = (chunk_paths == [wav_path])

    results = []
    try:
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Transcribing chunk {i + 1}/{len(chunk_paths)}")
            text = transcribe_chunk(chunk_path, recognizer)
            if text:
                results.append(text)
    finally:
        if not is_single_chunk:
            for p in chunk_paths:
                if p != wav_path and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    combined = " ".join(results).strip()
    if not combined:
        raise sr.UnknownValueError()

    return combined


# ─────────────────────────────────────────────
# 3. Translate  (Malayalam → English)
# ─────────────────────────────────────────────

def translate_to_english(text: str) -> str:
    """Translate Malayalam text to English via deep-translator."""
    if not text or not text.strip():
        return ""
    return GoogleTranslator(source="ml", target="en").translate(text) or ""


# ─────────────────────────────────────────────
# 4. Medical extraction via remote Ollama
# ─────────────────────────────────────────────
# The prompt lives in the Colab notebook (POST /extract).
# Here we just forward the Malayalam transcript and parse the response.

def extract_medical_info(transcript: str) -> tuple[Optional[MedicalExtraction], Optional[str]]:
    """
    POST { "transcript": "<malayalam text>" } to Colab /extract endpoint.
    The Colab notebook handles the Ollama prompt and JSON parsing internally.

    Returns:
        (MedicalExtraction, None)   — on success
        (None, error_message: str) — on failure
    """
    if not transcript or not transcript.strip():
        return None, "Empty transcript — nothing to extract"

    try:
        resp = requests.post(
            OLLAMA_API_URL,                          # → {COLAB_BASE_URL}/extract
            json={"transcript": transcript.strip()}, # Colab expects exactly this key
            timeout=OLLAMA_TIMEOUT,
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",  # skip ngrok interstitial page
            },
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return None, f"Ollama request timed out after {OLLAMA_TIMEOUT}s"
    except requests.exceptions.ConnectionError as exc:
        return None, f"Cannot reach Colab at {OLLAMA_API_URL}: {exc}"
    except requests.exceptions.HTTPError as exc:
        return None, f"Ollama HTTP error {resp.status_code}: {exc}"

    try:
        body = resp.json()
    except ValueError:
        return None, f"Non-JSON response from Colab: {resp.text[:500]}"

    # Colab /extract returns { "success": true, "extraction": { ... }, ... }
    if not body.get("success"):
        return None, f"Colab extraction failed: {body}"

    try:
        extraction = MedicalExtraction(**body["extraction"])
    except Exception as exc:
        return None, f"Schema validation error: {exc}"

    return extraction, None


# ─────────────────────────────────────────────
# 5. Pipeline orchestrator
# ─────────────────────────────────────────────

def run_pipeline(
    audio_bytes: bytes,
    filename:    str  = "audio.webm",
    translate:   bool = True,
    extract:     bool = False,   # NEW — set True to call Ollama extraction
) -> dict:
    """
    Full pipeline: raw audio bytes → result dict.

    Args:
        audio_bytes : raw bytes from the uploaded file
        filename    : original filename (used to pick the temp-file suffix)
        translate   : if True, translate Malayalam → English
        extract     : if True, send transcript to remote Ollama for medical extraction

    Returns:
        {
            "success"    : bool,
            "malayalam"  : str,
            "english"    : str,
            "extraction" : dict | None,   # present when extract=True
            "error"      : str | None
        }
    """
    suffix = os.path.splitext(filename)[1].lower() or ".webm"
    if suffix not in SUPPORTED_SUFFIXES:
        return {
            "success": False, "malayalam": "", "english": "",
            "extraction": None,
            "error": f"Unsupported format '{suffix}'. Supported: {', '.join(SUPPORTED_SUFFIXES)}",
        }

    tmp_path = wav_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        wav_path = convert_to_wav(tmp_path)

        # ── Step 2: Transcribe ────────────────────────────────────────────
        try:
            malayalam_text = transcribe(wav_path)
        except sr.UnknownValueError:
            return {
                "success": False, "malayalam": "", "english": "",
                "extraction": None,
                "error": "Speech not clear. Please speak closer to the microphone.",
            }
        except sr.RequestError as exc:
            return {
                "success": False, "malayalam": "", "english": "",
                "extraction": None,
                "error": f"Google Speech API error: {exc}",
            }

        # ── Step 3: Translate (non-fatal) ─────────────────────────────────
        english_text     = ""
        translation_error = None
        if translate:
            try:
                english_text = translate_to_english(malayalam_text)
            except Exception as exc:
                logger.error(f"Translation error: {exc}")
                translation_error = f"Translation failed: {exc}"

        # ── Step 4: Medical extraction via Ollama (non-fatal) ─────────────
        extraction_dict  = None
        extraction_error = None
        if extract:
            # Always send the Malayalam text — Colab's prompt handles mixed-language input
            medical_data, extraction_error = extract_medical_info(malayalam_text)
            if medical_data:
                extraction_dict = medical_data.model_dump()
            else:
                logger.error(f"Medical extraction failed: {extraction_error}")

        # Compose the combined error message (translation + extraction, both non-fatal)
        combined_error = " | ".join(filter(None, [translation_error, extraction_error])) or None

        return {
            "success"   : True,
            "malayalam" : malayalam_text,
            "english"   : english_text,
            "extraction": extraction_dict,
            "error"     : combined_error,
        }

    except Exception as exc:
        logger.exception("Audio processing error")
        return {
            "success"   : False,
            "malayalam" : "",
            "english"   : "",
            "extraction": None,
            "error"     : f"Processing error: {exc}",
        }

    finally:
        for path in (tmp_path, wav_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
