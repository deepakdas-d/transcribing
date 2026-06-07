# Transcribe API

This project provides a FastAPI endpoint to transcribe audio files using Google Cloud Speech-to-Text.

## Prerequisites

1.  **Python 3.8+**
2.  **Google Cloud Service Account**: Ensure you have a `service.json` file in the project root with the necessary permissions for Google Cloud Speech-to-Text.

## Installation

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirement.txt
    ```

## Running the Server

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

## API Endpoints

### 1. Transcribe Audio
- **Endpoint**: `/transcribe`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### Testing with Postman

1.  Open Postman and create a new **POST** request.
2.  Enter the URL: `http://127.0.0.1:8000/transcribe`
3.  Go to the **Body** tab.
4.  Select **form-data**.
5.  In the `Key` column, type `file`.
6.  Change the `Key` type from `Text` to `File` (hover over the key field to see the dropdown).
7.  In the `Value` column, click **Select Files** and choose a `.wav` or `.mp3` file.
8.  Click **Send**.

#### Sample Response
```json
{
    "success": true,
    "transcript": "Hello this is a sample transcription."
}
```

## Supported Languages
- Primary: Malayalam (`ml-IN`)
- Alternative: English (`en-IN`)
- Automatic punctuation is enabled.
