"""
audio_processor.py — Chunked transcription + remote Ollama medical extraction

Pipeline:
  1. convert_to_wav()           — ffmpeg: any format → 16kHz mono WAV
  2. split_wav_into_chunks()    — splits audio > CHUNK_SECONDS into overlapping slices
  3. transcribe_chunk_groq()    — Groq Whisper large-v3 (ml), parallel Malayalam + English
  4. llama_correct()            — Groq LLaMA 70B cross-check + structured correction
  5. transcribe()               — orchestrates 2+3+4, chunk-aware
  6. extract_medical_info()     — POST transcript → remote Colab Ollama → MedicalExtraction
  7. run_pipeline()             — top-level orchestrator used by all endpoints

Replaces:
  - Google Speech Recognition (sr.Recognizer / recognize_google)
  - deep_translator.GoogleTranslator

With:
  - Groq API  : Whisper large-v3  (transcription + translation, parallel threads)
  - Groq API  : LLaMA-3.3-70B    (cross-check, terminology correction, structuring)
"""

import logging
import math
import os
import subprocess
import tempfile
import wave
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from groq import Groq
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".wav", ".mp3", ".mp4", ".webm", ".ogg", ".m4a", ".flac"}

# ── Chunking ──────────────────────────────────────────────────────────────────
# Groq Whisper accepts up to 25 MB / ~10 min; keep chunks well under that.
CHUNK_SECONDS   = 60    # longer chunks → better context for Whisper
OVERLAP_SECONDS = 2     # overlap to avoid cutting mid-word at boundaries

# ── Groq models ───────────────────────────────────────────────────────────────
GROQ_WHISPER_MODEL = "whisper-large-v3"
GROQ_LLAMA_MODEL   = "llama-3.3-70b-versatile"

# ── Remote Ollama (Google Colab via ngrok) ────────────────────────────────────
COLAB_BASE_URL = os.getenv("COLAB_BASE_URL", "https://repair-dolly-dollop.ngrok-free.dev")
OLLAMA_API_URL = f"{COLAB_BASE_URL}/extract"
OLLAMA_MODEL   = "qwen3:8b"
OLLAMA_TIMEOUT = 120

# ── Dental / medical vocabulary hints ─────────────────────────────────────────
MALAYALAM_DENTAL_VOCAB = (
    "ദന്ത ചികിത്സ, പല്ല് വേദന, extraction, root canal, filling, scaling, "
    "Amoxicillin, Metronidazole, Ibuprofen, Paracetamol, ഗുളിക, ദിവസം മൂന്ന് നേരം, "
    "ഒരാഴ്ച, ഫോളോ അപ്പ്, X-ray, crown, bridge, implant, abscess, cavity, gum, "
    "nerve, pulp, orthodontic, braces, wisdom tooth, molar, incisor"
)

ENGLISH_DENTAL_VOCAB = (
    "Dental clinic, tooth pain, root canal treatment, tooth extraction, dental filling, "
    "scaling and polishing, Amoxicillin 500mg, Metronidazole 400mg, Ibuprofen 400mg, "
    "Paracetamol 500mg, three times a day, five days, follow up, dental X-ray, "
    "crown, bridge, implant, abscess, cavity, gum disease, pulp, orthodontic treatment"
)


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
    chief_complaints:           list[str]          = Field(default_factory=list)
    patient_reported_symptoms:  list[str]          = Field(default_factory=list)
    symptoms:                   list[str]          = Field(default_factory=list)
    past_conditions_mentioned:  list[str]          = Field(default_factory=list)
    conditions_mentioned:       list[str]          = Field(default_factory=list)
    medications_mentioned:      list[str]          = Field(default_factory=list)
    prescribed_medications:     list[MedicineItem] = Field(default_factory=list)
    body_parts_mentioned:       list[str]          = Field(default_factory=list)
    duration:                   list[str]          = Field(default_factory=list)
    severity:                   list[str]          = Field(default_factory=list)
    doctor_observations:        list[str]          = Field(default_factory=list)
    doctor_confirmed_diagnosis: list[str]          = Field(default_factory=list)
    advice:                     list[str]          = Field(default_factory=list)
    recommended_tests:          list[str]          = Field(default_factory=list)
    follow_up:                  list[str]          = Field(default_factory=list)
    risk_flags:                 list[str]          = Field(default_factory=list)
    uncertain_items:            list[str]          = Field(default_factory=list)


# ─────────────────────────────────────────────
# 1. Convert to WAV  (16kHz mono s16)
# ─────────────────────────────────────────────

def convert_to_wav(input_path: str) -> str:
    """
    Convert any browser audio format (webm/ogg/mp4/etc.) to 16kHz mono WAV via ffmpeg.
    Applies a lightweight filter chain: highpass noise removal + loudness normalisation.
    Returns path to the WAV file, or original path if conversion fails.
    """
    wav_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", "highpass=f=60,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11",
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
# 2. Split WAV into chunks
# ─────────────────────────────────────────────

def split_wav_into_chunks(wav_path: str, chunk_seconds: int = CHUNK_SECONDS) -> list[str]:
    """
    Split a WAV file into fixed-size overlapping chunks using ffmpeg.
    Returns [wav_path] unchanged if the audio is short enough for a single call.
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
# 3. Groq Whisper — single chunk (parallel)
# ─────────────────────────────────────────────

def _groq_transcribe_chunk(client: Groq, wav_path: str) -> str:
    """
    Thread A — Whisper in TRANSCRIPTION mode with Malayalam language hint.
    Returns the raw Malayalam transcript for one chunk, or "" on failure.
    """
    try:
        with open(wav_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model=GROQ_WHISPER_MODEL,
                file=f,
                language="ml",
                prompt=MALAYALAM_DENTAL_VOCAB,
                response_format="text",
            )
        return (response or "").strip()
    except Exception as exc:
        logger.error(f"Groq transcribe error on {wav_path}: {exc}")
        return ""


def _groq_translate_chunk(client: Groq, wav_path: str) -> str:
    """
    Thread B — Whisper in TRANSLATION mode (audio → English directly).
    Returns raw English text for one chunk, or "" on failure.
    This will have terminology errors that LLaMA corrects in step 4.
    """
    try:
        with open(wav_path, "rb") as f:
            response = client.audio.translations.create(
                model=GROQ_WHISPER_MODEL,
                file=f,
                prompt=ENGLISH_DENTAL_VOCAB,
                response_format="text",
            )
        return (response or "").strip()
    except Exception as exc:
        logger.error(f"Groq translate error on {wav_path}: {exc}")
        return ""


# ─────────────────────────────────────────────
# 4. LLaMA — cross-check & correct
# ─────────────────────────────────────────────

def _llama_correct(client: Groq, malayalam: str, raw_english: str) -> str:
    """
    Use LLaMA 70B to cross-reference the Malayalam transcript (ground truth)
    against the raw Whisper English translation, fix medical terminology errors,
    and return a clean structured clinical note.

    Falls back to raw_english if the LLaMA call fails.
    """
    if not malayalam and not raw_english:
        return ""

    system_prompt = (
        "You are a senior dental clinic scribe. "
        "You receive two inputs for the SAME consultation:\n"
        "  1. Malayalam transcript (ground truth — more accurate phonetically)\n"
        "  2. Raw English translation (may have medicine name / dosage errors)\n\n"
        "Your task:\n"
        "- Cross-reference both to resolve ambiguities.\n"
        "- Fix medicine names (e.g. 'Amoxycillin' → 'Amoxicillin', "
        "'Metronidazole' spellings, dosage numbers).\n"
        "- Fix treatment terms (extraction, root canal, scaling, filling, crown, etc.).\n"
        "- Output ONLY a structured clinical note in exactly this format "
        "(omit sections with no data):\n\n"
        "Chief Complaint: <one line>\n"
        "Treatment Done: <one line>\n"
        "Medicines:\n"
        "  - <Name> <dose>, <frequency>, <duration>\n"
        "Instructions: <one line>\n"
        "Follow-up: <one line>\n\n"
        "Do NOT add any preamble, explanation, or extra commentary."
    )

    user_prompt = (
        f"Malayalam transcript:\n{malayalam}\n\n"
        f"Raw English translation:\n{raw_english}"
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_LLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,    # low temp → deterministic medical output
            max_tokens=512,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error(f"LLaMA correction error: {exc}")
        return raw_english  # non-fatal: fall back to uncorrected Whisper translation


# ─────────────────────────────────────────────
# 5. Transcribe full audio (chunk-aware, Groq)
# ─────────────────────────────────────────────

def transcribe(wav_path: str) -> tuple[str, str]:
    """
    Transcribe a WAV file using the Groq pipeline:
      - Splits long audio into CHUNK_SECONDS chunks
      - For each chunk, fires Thread A (Malayalam) and Thread B (English) in parallel
      - After all chunks, feeds combined Malayalam + combined raw English into LLaMA

    Returns:
        (malayalam_text: str, corrected_english: str)

    Raises:
        ValueError  — GROQ_API_KEY not set
        RuntimeError — all chunks returned empty (no speech detected)
        Exception   — Groq API / network error propagated to run_pipeline
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)

    chunk_paths     = split_wav_into_chunks(wav_path, chunk_seconds=CHUNK_SECONDS)
    is_single_chunk = (chunk_paths == [wav_path])

    malayalam_parts: list[str] = []
    raw_english_parts: list[str] = []

    try:
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Processing chunk {i + 1}/{len(chunk_paths)} with Groq Whisper (parallel)")

            # Fire both Whisper modes in parallel for this chunk
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_ml = pool.submit(_groq_transcribe_chunk, client, chunk_path)
                future_en = pool.submit(_groq_translate_chunk,  client, chunk_path)

                ml_text = future_ml.result()
                en_text = future_en.result()

            if ml_text:
                malayalam_parts.append(ml_text)
            if en_text:
                raw_english_parts.append(en_text)

            logger.info(
                f"Chunk {i + 1}: Malayalam={'yes' if ml_text else 'empty'}, "
                f"RawEnglish={'yes' if en_text else 'empty'}"
            )

    finally:
        # Clean up temp chunk files (never delete the original wav_path)
        if not is_single_chunk:
            for p in chunk_paths:
                if p != wav_path and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    combined_malayalam = " ".join(malayalam_parts).strip()
    combined_raw_en    = " ".join(raw_english_parts).strip()

    if not combined_malayalam and not combined_raw_en:
        raise RuntimeError("No speech detected in audio. Please speak clearly and closer to the microphone.")

    # LLaMA cross-check + structuring (uses combined output of all chunks)
    logger.info("Running LLaMA cross-check and correction")
    corrected_english = _llama_correct(client, combined_malayalam, combined_raw_en)

    return combined_malayalam, corrected_english


# ─────────────────────────────────────────────
# 6. Medical extraction via remote Ollama
# ─────────────────────────────────────────────

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
            OLLAMA_API_URL,
            json={"transcript": transcript.strip()},
            timeout=OLLAMA_TIMEOUT,
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
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

    if not body.get("success"):
        return None, f"Colab extraction failed: {body}"

    try:
        extraction = MedicalExtraction(**body["extraction"])
    except Exception as exc:
        return None, f"Schema validation error: {exc}"

    return extraction, None


# ─────────────────────────────────────────────
# 7. Pipeline orchestrator
# ─────────────────────────────────────────────

def run_pipeline(
    audio_bytes: bytes,
    filename:    str  = "audio.webm",
    translate:   bool = True,
    extract:     bool = False,
) -> dict:
    """
    Full pipeline: raw audio bytes → result dict.

    Args:
        audio_bytes : raw bytes from the uploaded file
        filename    : original filename (used to pick the temp-file suffix)
        translate   : if True, produce an English translation (always True now —
                      Groq Whisper always generates English alongside Malayalam;
                      this flag is kept for API compatibility)
        extract     : if True, send transcript to remote Ollama for medical extraction

    Returns:
        {
            "success"    : bool,
            "malayalam"  : str,
            "english"    : str,   ← LLaMA-corrected structured clinical note
            "extraction" : dict | None,
            "error"      : str | None
        }

    Note on translate=False:
        The Groq pipeline always runs both Whisper modes in parallel (it costs
        nothing extra). When translate=False the 'english' field is still
        populated but contains only the raw Whisper translation (LLaMA correction
        is skipped to save latency when the caller doesn't need structured output).
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

        # ── Step 3+4: Transcribe (parallel Groq Whisper) + LLaMA correction ──
        try:
            malayalam_text, english_text = transcribe(wav_path)
        except ValueError as exc:
            # Missing API key — hard failure
            return {
                "success": False, "malayalam": "", "english": "",
                "extraction": None, "error": str(exc),
            }
        except RuntimeError as exc:
            # No speech detected — hard failure
            return {
                "success": False, "malayalam": "", "english": "",
                "extraction": None, "error": str(exc),
            }
        except Exception as exc:
            # Groq API / network error — hard failure
            return {
                "success": False, "malayalam": "", "english": "",
                "extraction": None,
                "error": f"Groq API error: {exc}",
            }

        # ── Step 5: Medical extraction via Ollama (non-fatal) ────────────────
        extraction_dict  = None
        extraction_error = None
        if extract:
            medical_data, extraction_error = extract_medical_info(malayalam_text)
            if medical_data:
                extraction_dict = medical_data.model_dump()
            else:
                logger.error(f"Medical extraction failed: {extraction_error}")

        combined_error = extraction_error or None

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