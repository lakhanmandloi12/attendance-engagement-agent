import io
import os
import json
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
client = (
    genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=50000))
    if API_KEY
    else None
)
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]

RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

AUDIO_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
               "ogg": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac"}
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


class Problem(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


# ---------- Epic 2: vision (face detection + attendance) ----------
def detect_faces(img):
    """Find faces with OpenCV and draw a box around each one."""
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    out = arr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (15, 157, 138), 3)
    return len(faces), Image.fromarray(out)


# ---------- Epic 3 + 4: audio and multimodal AI ----------
def build_prompt(has_audio):
    audio = (
        '"audio": {"speech_percent": <0-100>, "silence_percent": <0-100>, '
        '"noise_level": "Low"|"Medium"|"High", "audio_score": <0-100>, '
        '"summary": "<one sentence>"},'
        if has_audio else '"audio": null,'
    )
    extra = " and the class audio recording" if has_audio else ""
    return f"""You are a classroom analytics assistant. Analyze the classroom photo{extra}.
Reply ONLY with JSON in this exact shape:
{{
  "people_count": <integer, people visible in the photo>,
  "vision_score": <integer 0-100, engagement from posture, gaze and activity>,
  "reason": "<one short sentence>",
  {audio}
  "suggestions": ["<tip for the teacher>", "<tip for the teacher>", "<tip for the teacher>"]
}}
If the photo has no people, use people_count 0 and vision_score 0 and say so in reason."""


def ask_gemini(parts):
    if client is None:
        raise Problem("API key not found. Check GEMINI_API_KEY in .env and restart the server.", 500)
    started, use_fast, last = time.time(), True, "unknown"
    for model in MODELS:
        for _ in range(2):
            if time.time() - started > 50:
                break
            try:
                kwargs = {"response_mime_type": "application/json"}
                if use_fast:
                    kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                r = client.models.generate_content(
                    model=model, contents=parts,
                    config=types.GenerateContentConfig(**kwargs),
                )
                print("OK", model)
                return json.loads(r.text)
            except Exception as e:
                code = getattr(e, "code", None)
                last = f"{model}: {e}"
                print(model, "failed:", repr(e))
                if code in (401, 403) or (code == 400 and "api key" in str(e).lower()):
                    raise Problem(f"API key problem: {e}", 500)
                if code == 400 and use_fast:
                    use_fast = False
                    continue
                if code in (400, 404, 504):
                    break
                time.sleep(2)
    raise Problem(f"The AI is not responding right now. Try again in a minute. ({last})", 503)


def clamp(v, default=0):
    try:
        return max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return default


def level_of(score):
    return "High" if score >= 70 else "Medium" if score >= 40 else "Low"


# ---------- Epic 5: session history ----------
def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_history(entry):
    history = [entry] + load_history()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[:50], f)


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/history")
def history():
    return jsonify({"sessions": load_history()})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        photo = request.files.get("photo")
        if not photo or not photo.filename:
            raise Problem("Choose a classroom photo first.")
        try:
            img = Image.open(photo.stream).convert("RGB")
        except Exception:
            raise Problem("That file is not a valid image. Upload a JPG or PNG.")
        img.thumbnail((1280, 1280))

        raw_total = request.form.get("total_students", "").strip()
        total = int(raw_total) if raw_total.isdigit() and int(raw_total) > 0 else None

        audio_part = None
        audio = request.files.get("audio")
        if audio and audio.filename:
            ext = audio.filename.rsplit(".", 1)[-1].lower()
            if ext not in AUDIO_TYPES:
                raise Problem("Audio must be MP3, WAV, M4A, OGG, FLAC or AAC.")
            data = audio.read()
            if not data:
                raise Problem("The audio file is empty.")
            audio_part = types.Part.from_bytes(data=data, mime_type=AUDIO_TYPES[ext])

        faces, boxed = detect_faces(img)
        rid = uuid.uuid4().hex[:10]
        boxed.save(os.path.join(RESULTS_DIR, rid + ".jpg"), "JPEG", quality=85)

        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        parts = [types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")]
        if audio_part:
            parts.append(audio_part)
        parts.append(build_prompt(audio_part is not None))
        ai = ask_gemini(parts)

        # attendance: best of OpenCV faces and AI estimate, capped by class size
        ai_count = clamp(ai.get("people_count"), 0) if str(ai.get("people_count", "")).isdigit() else 0
        present = max(faces, ai_count)
        if total:
            present = min(present, total)
        pct = round(present / total * 100) if total else None

        # fusion: 60% vision + 40% audio (vision only when no audio)
        vision = clamp(ai.get("vision_score"))
        a = ai.get("audio") if audio_part else None
        audio_info, audio_score = None, None
        if isinstance(a, dict):
            audio_score = clamp(a.get("audio_score"))
            audio_info = {
                "speech_percent": clamp(a.get("speech_percent")),
                "silence_percent": clamp(a.get("silence_percent")),
                "noise_level": a.get("noise_level") if a.get("noise_level") in ("Low", "Medium", "High") else "Medium",
                "audio_score": audio_score,
                "summary": str(a.get("summary", "")),
            }
        final = round(0.6 * vision + 0.4 * audio_score) if audio_score is not None else vision

        result = {
            "image_url": f"/static/results/{rid}.jpg",
            "faces": faces, "present": present, "total": total, "attendance_pct": pct,
            "vision_score": vision, "audio": audio_info,
            "final_score": final, "level": level_of(final),
            "reason": str(ai.get("reason", "")),
            "suggestions": [str(s) for s in (ai.get("suggestions") or [])][:3],
        }
        save_history({
            "time": datetime.now().strftime("%d %b, %I:%M %p"),
            "present": present, "total": total, "attendance_pct": pct,
            "final_score": final, "level": result["level"], "audio": audio_info is not None,
        })
        return jsonify(result)
    except Problem as p:
        return jsonify({"error": p.message}), p.status
    except Exception as e:
        print("UNEXPECTED:", repr(e))
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.errorhandler(413)
def too_big(_):
    return jsonify({"error": "Files are too large. Keep photo and audio under 16 MB in total."}), 413


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)