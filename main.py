from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv
import tempfile
import requests
import os

load_dotenv()

app = FastAPI()

COLAB_WHISPER_URL = os.getenv("COLAB_WHISPER_URL")

if not COLAB_WHISPER_URL:
    raise Exception("COLAB_WHISPER_URL missing from .env")


@app.get("/")
def health():
    return {
        "status": "ok",
        "whisper_server": COLAB_WHISPER_URL
    }


@app.post("/google-transcribe")
async def google_transcribe(
    file: UploadFile = File(...)
):
    temp_path = None

    try:
        suffix = (
            os.path.splitext(file.filename)[1]
            if file.filename
            else ".wav"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_path = tmp.name

        with open(temp_path, "rb") as audio_file:

            files = {
                "file": (
                    file.filename,
                    audio_file,
                    file.content_type
                )
            }

            response = requests.post(
                f"{COLAB_WHISPER_URL}/transcribe",
                files=files,
                timeout=1800
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        whisper_response = response.json()

        return {
            "success": True,
            "engine": "whisper-large-v3",
            "source": COLAB_WHISPER_URL,
            "result": whisper_response
        }

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Whisper server timeout"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )