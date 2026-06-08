from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tempfile
import requests
import os

load_dotenv()

app = FastAPI(title="Whisper Transcription API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLAB_WHISPER_URL = os.getenv("COLAB_WHISPER_URL")

if not COLAB_WHISPER_URL:
    raise Exception("COLAB_WHISPER_URL missing from .env")

SUPPORTED_FORMATS = {".wav", ".mp3", ".mp4", ".webm", ".ogg", ".m4a", ".flac"}


@app.get("/")
def health():
    return {
        "status": "ok",
        "whisper_server": COLAB_WHISPER_URL,
        "engine": "whisper-large-v3",
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file and receive a Malayalam transcription from Whisper.
    Supported formats: wav, mp3, mp4, webm, ogg, m4a, flac
    """
    suffix = (
        os.path.splitext(file.filename)[1].lower()
        if file.filename
        else ".wav"
    )

    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{suffix}'. Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        with open(temp_path, "rb") as audio_file:
            response = requests.post(
                f"{COLAB_WHISPER_URL}/transcribe",
                files={"file": (file.filename, audio_file, file.content_type)},
                timeout=1800,
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
            )

        result = response.json()

        return {
            "success": True,
            "engine": "whisper-large-v3",
            "source": COLAB_WHISPER_URL,
            "filename": file.filename,
            "result": result,
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Whisper server timed out")

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)