import io
import os
import json
import time
import mimetypes
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types

try:
    from PIL import Image
except ImportError:
    Image = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
client = (
    genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=40000))
    if API_KEY
    else None
)

MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]

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


def shrink(image_bytes, mime):
    """Resize big photos to max 1024px so upload and AI processing are faster."""
    if Image is None:
        return image_bytes, mime
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print("resize skipped:", e)
        return image_bytes, mime


def make_config(fast):
    kwargs = {"response_mime_type": "application/json"}
    if fast:
        # turn off extra "thinking" time; this is a simple vision task
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


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
        image_bytes, mime = shrink(f.read(), mime)

    last_error = "unknown"
    started = time.time()
    use_fast = True
    for model in MODELS:
        for attempt in range(2):
            if time.time() - started > 40:
                break
            t0 = time.time()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        PROMPT,
                    ],
                    config=make_config(use_fast),
                )
                print(f"OK {model} in {time.time() - t0:.1f}s")
                try:
                    return jsonify({"report": json.loads(response.text)})
                except (ValueError, TypeError):
                    return jsonify({"raw": response.text})
            except Exception as e:
                code = getattr(e, "code", None)
                last_error = f"{model}: {e}"
                print(f"{model} attempt {attempt + 1} failed after {time.time() - t0:.1f}s:", repr(e))
                if code in (401, 403) or (code == 400 and "api key" in str(e).lower()):
                    return jsonify({"error": f"API key problem: {e}"}), 500
                if code == 400:
                    if use_fast:
                        use_fast = False  # extra setting not accepted: retry without it
                        continue
                    break  # this model rejects the request: try the next model
                if code in (404, 504):
                    break  # model missing or too slow: try the next model
                time.sleep(2)

    return jsonify({"error": f"The AI is not responding right now. Try again in a minute. ({last_error})"}), 503


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)