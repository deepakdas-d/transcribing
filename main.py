from fastapi import FastAPI, UploadFile, File, HTTPException
from google.cloud import speech
import tempfile
import os
import subprocess
import json

app = FastAPI()

# Initialize once with the service account file
SERVICE_ACCOUNT_FILE = "service.json"

if not os.path.exists(SERVICE_ACCOUNT_FILE):
    SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "service.json")

try:
    # We try to initialize the client
    client = speech.SpeechClient.from_service_account_file(SERVICE_ACCOUNT_FILE)
except Exception as e:
    print(f"Error loading service account: {e}")
    client = None

def convert_audio_to_wav(input_path, output_path):
    """Convert any audio to 16kHz mono PCM WAV."""
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts+igndts",   # ignore bad timestamps
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-vn",                          # ignore any video stream
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("FFMPEG ERROR:")
            print(result.stderr)
            return False
        return True

    except Exception as e:
        print(f"ffmpeg conversion error: {e}")
        return False
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Speech client not initialized. Ensure service.json is present."
        )

    temp_path = None
    converted_path = None
    try:
        # Read uploaded file
        contents = await file.read()

        # Save temporarily to detect extension and use ffprobe
        filename = file.filename or "audio.wav"
        extension = os.path.splitext(filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension or ".wav"
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # Check original duration
        orig_check = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", temp_path],
            capture_output=True, text=True
        )
        print("Original duration:", orig_check.stdout)
        # Convert to 16kHz mono WAV
        converted_path = temp_path + "_converted.wav"
        if not convert_audio_to_wav(temp_path, converted_path):
            raise Exception("Failed to convert audio file to WAV format.")

        # ✅ Add here — check converted file duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", converted_path],
            capture_output=True, text=True
        )   
        print("Converted duration:", result.stdout)

        # Read audio content for processing
        with open(converted_path, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)

        # Always use LINEAR16, 16000 Hz, mono
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            audio_channel_count=1,

            language_code="ml-IN",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
        )

        print(f"Starting long_running_recognize for {filename} (Converted to 16kHz WAV)")

        # Long running recognition
        operation = client.long_running_recognize(
            config=config,
            audio=audio
        )

        print("Waiting for operation to complete...")
        response = operation.result(timeout=600)

        transcript_parts = []
        last_word_end = 0.0
        for result in response.results:
            if result.alternatives:
                transcript_parts.append(result.alternatives[0].transcript)
                # Track last word end time to check coverage
                for word in result.alternatives[0].words:
                    end = word.end_time.total_seconds()
                    if end > last_word_end:
                        last_word_end = end

        print(f"Audio covered up to: {last_word_end:.2f}s")

        full_transcript = " ".join(transcript_parts).strip()

        if not full_transcript:
            return {
                "success": False,
                "detail": "Google returned no transcript.",
                "segments_count": len(response.results),
                "google_response": str(response)
            }

        return {
            "success": True,
            "transcript": full_transcript,
            "segments_count": len(response.results),
            "detected_info": {"conversion": "16kHz mono WAV"}
        }

    except Exception as e:
        error_msg = str(e)
        print(f"Transcription error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    finally:
        for path in [temp_path, converted_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
