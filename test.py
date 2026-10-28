import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

key = (os.getenv("GEMINI_API_KEY") or "").strip()
print("Key mili:", bool(key), "| shuru ke 3 akshar:", key[:3], "| length:", len(key))

client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000))

for m in ["gemini-3.8-flash", "gemini-3-flash-preview", "gemini-flash-latest"]:
    try:
        r = client.models.generate_content(model=m, contents="Say hello in one word")
        print("OK  ", m, "->", r.text)
        break
    except Exception as e:
        print("FAIL", m, "->", e)