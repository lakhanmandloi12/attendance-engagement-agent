# Attendance & Engagement Agent

A Flask web app that takes a classroom photo (and optional class audio) and reports attendance and engagement using OpenCV face detection and Google Gemini.

## Workflow
1. Teacher uploads a classroom photo, optional audio, and the class size.
2. **Vision:** OpenCV detects faces and draws boxes; Gemini estimates headcount and engagement from posture and gaze.
3. **Audio:** Gemini measures speech time, silence, noise level and gives an audio engagement score.
4. **Fusion:** final engagement = 60% vision + 40% audio (vision only if no audio).
5. **Attendance:** present = best of face count and AI estimate, capped at class size; attendance % = present / enrolled.
6. **Analytics:** the report shows suggestions for the teacher and each session is saved to a History tab.

## Setup
1. `pip install -r requirements.txt`
2. Create `.env` with `GEMINI_API_KEY=your_key`
3. `python app.py` and open http://127.0.0.1:5000

## Edge cases handled
Invalid or missing image, unsupported or empty audio, files over 16 MB, photos with no people, missing API key, busy or slow AI models (retries and fallback models).

## Limitations
OpenCV's Haar face detector finds mainly front-facing faces, so Gemini's headcount is used as a cross-check. Engagement is an AI estimate, not a measurement.

## Conclusion
The agent turns a photo and a recording into an attendance figure, an engagement score and practical teaching suggestions, replacing manual roll calls with a quick, privacy-conscious check.