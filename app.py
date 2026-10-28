import os
import json
import time
import mimetypes
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
client = (
    genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=60000))
    if API_KEY
    else None
)

MODELS = ["gemini-3.8-flash", "gemini-3-flash-preview", "gemini-flash-latest"]

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PROMPT = """You are a classroom analytics assistant.
Analyze this classroom image and reply ONLY with JSON in this exact shape:
{
  "people_count": <integer, approximate number of people visible>,
  "engagement": "High" | "Medium" | "Low",
  "engagement_score": <integer 0-100>,
  "reason": "<one short sentence based on posture, gaze and activity>",
  "suggestions": ["<short tip for the teacher>", "<short tip for the teacher>"]
}
If there are no people in the image, use people_count 0, engagement "Low",
engagement_score 0, and explain that in "reason"."""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if client is None:
        return jsonify({"error": "API key not found. Check GEMINI_API_KEY in .env and restart the server."}), 500

    file = request.files.get("photo")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    mime = file.mimetype or mimetypes.guess_type(filename)[0] or "image/jpeg"
    if not mime.startswith("image/"):
        return jsonify({"error": "Please upload an image file (JPG or PNG)."}), 400

    with open(path, "rb") as f:
        image_bytes = f.read()

    last_error = "unknown"
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
                try:
                    report = json.loads(response.text)
                    return jsonify({"report": report})
                except (ValueError, TypeError):
                    return jsonify({"raw": response.text})
            except Exception as e:
                code = getattr(e, "code", None)
                last_error = f"{model}: {e}"
                print(f"{model} attempt {attempt + 1} failed:", repr(e))
                if code in (400, 401, 403):
                    return jsonify({"error": f"API key or request problem: {e}"}), 500
                if code == 404:
                    break
                time.sleep(2 * (attempt + 1))

    return jsonify({"error": f"The AI is not responding right now. Try again in a minute. ({last_error})"}), 503


if __name__ == "__main__":
    app.run(debug=True)