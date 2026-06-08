"""
main.py — Audio Transcription & Translation Service

Endpoints:
  GET  /                         — health check
  POST /transcribe               — audio → Malayalam text  (Google Speech ml-IN)
  POST /transcribe-and-translate — audio → Malayalam + English translation
"""

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from audio_processor import run_pipeline

load_dotenv()

app = FastAPI(
    title="Transcription & Translation API",
    description=(
        "Upload audio → Malayalam via Google Speech Recognition (ml-IN), "
        "optional Malayalam→English via Google Translate."
    ),
    version="2.0.0",
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
        "status": "ok",
        "stt_engine": "google-speech-recognition",
        "stt_language": "ml-IN",
        "translation": "deep-translator (Google)",
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
        { "detail": "Speech not clear..." | "Google Speech API error: ..." }
    """
    audio_bytes = await file.read()
    result = run_pipeline(audio_bytes, filename=file.filename or "audio.webm", translate=False)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success"  : True,
        "filename" : file.filename,
        "malayalam": result["malayalam"],
    }


# ─────────────────────────────────────────────
# POST /transcribe-and-translate
# Audio → Malayalam + English
# ─────────────────────────────────────────────

@app.post("/transcribe-and-translate", status_code=status.HTTP_200_OK, tags=["pipeline"])
async def transcribe_and_translate(file: UploadFile = File(...)):
    """
    Upload audio, receive Malayalam transcription AND English translation.

    FormData:
        file — audio blob (wav / mp3 / mp4 / webm / ogg / m4a / flac)

    Response 200:
        {
            "success"  : true,
            "filename" : "...",
            "malayalam": "...",
            "english"  : "..."   ← empty string if translation failed (non-fatal)
        }

    Response 400:
        { "detail": "Speech not clear..." | "Google Speech API error: ..." }
    """
    audio_bytes = await file.read()
    result = run_pipeline(audio_bytes, filename=file.filename or "audio.webm", translate=True)

    # Hard failure (STT failed) — return 400
    if not result["success"] and not result["malayalam"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success"  : True,
        "filename" : file.filename,
        "malayalam": result["malayalam"],
        "english"  : result["english"],
        # Surface translation error in response if it happened but was non-fatal
        **({"translation_error": result["error"]} if result["error"] else {}),
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)