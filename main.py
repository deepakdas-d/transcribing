"""
main.py — Audio Transcription, Translation & Medical Extraction Service

Endpoints:
  GET  /                              — health check
  POST /transcribe                    — audio → Malayalam text  (Groq Whisper ml)
  POST /transcribe-and-translate      — audio → Malayalam + structured English (Groq Whisper + LLaMA)
  POST /transcribe-and-extract        — audio → Malayalam + English + MedicalExtraction (Ollama)
"""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from audio_processor import run_pipeline, COLAB_BASE_URL, OLLAMA_MODEL, GROQ_WHISPER_MODEL, GROQ_LLAMA_MODEL

load_dotenv()

app = FastAPI(
    title="Transcription, Translation & Medical Extraction API",
    description=(
        "Upload audio → Malayalam via Groq Whisper large-v3 (ml), "
        "Malayalam→English via Groq Whisper translation mode, "
        "LLaMA 70B cross-check & clinical note structuring, "
        "optional structured medical extraction via remote Ollama (Colab)."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/", tags=["health"])
def health():
    return {
        "status"         : "ok",
        "version"        : "4.0.0",
        "stt_engine"     : f"groq-{GROQ_WHISPER_MODEL}",
        "stt_language"   : "ml (Malayalam)",
        "translation"    : f"groq-{GROQ_WHISPER_MODEL} (translate mode)",
        "correction"     : f"groq-{GROQ_LLAMA_MODEL} (cross-check + structuring)",
        "extraction"     : f"Ollama {OLLAMA_MODEL} via {COLAB_BASE_URL}",
    }


# ─────────────────────────────────────────────
# POST /transcribe
# Audio → Malayalam text only
# ─────────────────────────────────────────────

@app.post("/transcribe", status_code=status.HTTP_200_OK, tags=["pipeline"])
async def transcribe_only(file: UploadFile = File(...)):
    """
    Upload audio, receive Malayalam transcription only.

    FormData:
        file — audio blob (wav / mp3 / mp4 / webm / ogg / m4a / flac)

    Response 200:
        { "success": true, "filename": "...", "malayalam": "..." }

    Response 400:
        { "detail": "No speech detected..." | "Groq API error: ..." }
    """
    audio_bytes = await file.read()
    result = run_pipeline(
        audio_bytes,
        filename=file.filename or "audio.webm",
        translate=False,
        extract=False,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success"  : True,
        "filename" : file.filename,
        "malayalam": result["malayalam"],
    }


# ─────────────────────────────────────────────
# POST /transcribe-and-translate
# Audio → Malayalam + structured English clinical note
# ─────────────────────────────────────────────

@app.post("/transcribe-and-translate", status_code=status.HTTP_200_OK, tags=["pipeline"])
async def transcribe_and_translate(file: UploadFile = File(...)):
    """
    Upload audio, receive Malayalam transcription AND LLaMA-corrected English
    structured clinical note.

    FormData:
        file — audio blob (wav / mp3 / mp4 / webm / ogg / m4a / flac)

    Response 200:
        {
            "success"  : true,
            "filename" : "...",
            "malayalam": "...",
            "english"  : "Chief Complaint: ...\\nTreatment Done: ...\\n..."
        }

    Response 400:
        { "detail": "No speech detected..." | "Groq API error: ..." }
    """
    audio_bytes = await file.read()
    result = run_pipeline(
        audio_bytes,
        filename=file.filename or "audio.webm",
        translate=True,
        extract=False,
    )

    if not result["success"] and not result["malayalam"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success"  : True,
        "filename" : file.filename,
        "malayalam": result["malayalam"],
        "english"  : result["english"],
        **({"pipeline_error": result["error"]} if result["error"] else {}),
    }


# ─────────────────────────────────────────────
# POST /transcribe-and-extract
# Audio → Malayalam + English + MedicalExtraction
# ─────────────────────────────────────────────

@app.post("/transcribe-and-extract", status_code=status.HTTP_200_OK, tags=["pipeline"])
async def transcribe_and_extract(file: UploadFile = File(...)):
    """
    Upload audio, receive Malayalam transcription, LLaMA-corrected English clinical
    note, AND a fully structured MedicalExtraction object from remote Ollama.

    FormData:
        file — audio blob (wav / mp3 / mp4 / webm / ogg / m4a / flac)

    Response 200:
        {
            "success"    : true,
            "filename"   : "...",
            "malayalam"  : "...",
            "english"    : "...",
            "extraction" : {
                "chief_complaints"           : [...],
                "patient_reported_symptoms"  : [...],
                "symptoms"                   : [...],
                "past_conditions_mentioned"  : [...],
                "conditions_mentioned"       : [...],
                "medications_mentioned"      : [...],
                "prescribed_medications"     : [
                    { "name": "...", "dose": "...", "frequency": "...",
                      "duration": "...", "instructions": "..." }
                ],
                "body_parts_mentioned"       : [...],
                "duration"                   : [...],
                "severity"                   : [...],
                "doctor_observations"        : [...],
                "doctor_confirmed_diagnosis" : [...],
                "advice"                     : [...],
                "recommended_tests"          : [...],
                "follow_up"                  : [...],
                "risk_flags"                 : [...],
                "uncertain_items"            : [...]
            }
        }

    Response 400:
        { "detail": "No speech detected..." | "Groq API error: ..." }

    Notes:
        - Extraction failure is non-fatal; extraction field will be null with an
          "extraction_error" key explaining what went wrong.
    """
    audio_bytes = await file.read()
    result = run_pipeline(
        audio_bytes,
        filename=file.filename or "audio.webm",
        translate=True,
        extract=True,
    )

    # Hard failure — STT completely failed
    if not result["success"] and not result["malayalam"]:
        raise HTTPException(status_code=400, detail=result["error"])

    response: dict = {
        "success"   : True,
        "filename"  : file.filename,
        "malayalam" : result["malayalam"],
        "english"   : result["english"],
        "extraction": result["extraction"],   # None if Ollama call failed
    }

    # Surface non-fatal errors (extraction) without breaking 200
    if result["error"]:
        errors = [e.strip() for e in result["error"].split("|") if e.strip()]
        for err in errors:
            if "ollama" in err.lower() or "extraction" in err.lower() or "colab" in err.lower():
                response["extraction_error"] = err
            else:
                response["pipeline_error"] = err

    return response


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)