# Author: Bilal Saifi
# Version: 2.5 - Streamlit Cloud Compatible (No pydub)
# Notes: for more audios https://cloud.google.com/text-to-speech/docs/list-voices-and-types
# To execute this code run: streamlit run audio-book.py

import os
import re
import html
import tempfile
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from google.cloud import texttospeech, aiplatform
from vertexai.preview.generative_models import GenerativeModel
import streamlit as st

# === Google Cloud Config (read from secrets.toml on Streamlit Cloud) ===
project_id = "staging-sparkl-me"
location = "us-central1"
aiplatform.init(project=project_id, location=location)

# === Utility Functions ===
def extract_text(path):
    if path.endswith(".pdf"):
        pages = convert_from_path(path)
        text = ""
        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page)
            if len(page_text.strip()) > 20:
                text += f"\n[Page {i+1}]\n{page_text}"
        return text
    elif path.endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    elif path.endswith(".txt"):
        with open(path, "r") as f:
            return f.read()
    return ""

def clean_text(text):
    text = re.sub(r"\[Page \d+\]", "", text)
    text = re.sub(r"[_*~`]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def sanitize_ssml(text):
    text = html.unescape(text.replace("&", "&amp;"))
    text = re.sub(r'<(\w+)(\s[^>]*)?>', r'<\1>', text)
    if not text.strip().startswith("<speak>"):
        text = f"<speak>{text}</speak>"
    return text

def generate_teaching_script(raw_text, language_mode, prompt_override):
    model = GenerativeModel("gemini-2.5-pro-preview-05-06")
    prompt = None

    if prompt_override:
        prompt = f"{prompt_override.strip()}\n\nContent:\n{raw_text}"
    elif language_mode == "hinglish":
        prompt = f"""
        Aap ek friendly Indian teacher hain. Niche diye gaye content ko students ke liye simple Hinglish mein explain kijiye —
        Hindi aur English ka mix use karke.

        ✅ Jargon avoid kijiye, lekin technical terms English mein rakhiye
        ✅ Tone bilkul friendly aur classroom jaise ho
        ✅ Easy examples aur short sentences use kijiye
        ✅ Avoid unsupported SSML tags.
        ✅ Use SSML tags if supported (<speak>, <break>, <prosody>, etc.) and do not use '*'
        ✅ Voice narration ke liye suitable script likhiye

        Content:
        {raw_text}
        """
    else:
        prompt = f"""
        You are an audiobook narrator and teacher. Summarize and explain the following educational content in an easy, short, student-friendly script.

        ✅ Make it sound natural and teacher-like.
        ✅ Use SSML tags if supported (<speak>, <break>, <prosody>, etc.)
        ✅ Avoid unsupported SSML tags.

        Content:
        {raw_text}
        """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return ""

def split_by_bytes(text, max_bytes=4400):
    parts, current = [], ""
    for line in text.splitlines():
        test = current + line + "\n"
        if len(test.encode("utf-8")) <= max_bytes:
            current = test
        else:
            if current.strip():
                parts.append(current.strip())
            current = line + "\n"
    if current.strip():
        parts.append(current.strip())
    return parts

def generate_audio_chunks(script, voice_name, language_code, speaking_rate, pitch, max_bytes, use_rate, use_pitch):
    client = texttospeech.TextToSpeechClient()
    supports_ssml = not any(x in voice_name.lower() for x in ["chirp", "neural2", "hd", "casual"])
    chunks = split_by_bytes(script, max_bytes=max_bytes)
    audio_chunks = []

    for i, chunk in enumerate(chunks):
        plain_text = re.sub(r"<[^>]+>", "", chunk)
        try:
            if supports_ssml:
                input_text = texttospeech.SynthesisInput(ssml=sanitize_ssml(chunk))
            else:
                input_text = texttospeech.SynthesisInput(text=plain_text)
        except Exception:
            input_text = texttospeech.SynthesisInput(text=plain_text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )

        audio_config_params = {
            "audio_encoding": texttospeech.AudioEncoding.MP3
        }
        if use_rate:
            audio_config_params["speaking_rate"] = speaking_rate
        if use_pitch:
            audio_config_params["pitch"] = pitch

        audio_config = texttospeech.AudioConfig(**audio_config_params)

        try:
            response = client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config,
            )
            audio_chunks.append(response.audio_content)
        except Exception as e:
            st.warning(f"Chunk {i+1} failed: {e}")

    if audio_chunks:
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_mp3.name, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
        return temp_mp3.name
    return None

# === Streamlit UI ===
st.set_page_config(page_title="AI Audiobook Generator", layout="wide")
st.title("🎙️ AI-Powered Audiobook Generator")

uploaded_file = st.file_uploader("📂 Upload PDF, Image, or Text File", type=["pdf", "png", "jpg", "jpeg", "txt"])

col1, col2 = st.columns(2)
with col1:
    language_mode = st.selectbox("🗣️ Language Style", ["english", "hinglish"])
    language_code = st.text_input("🌐 TTS Language Code", "en-US")
    voice_name = st.text_input("🎙️ TTS Voice Name", "en-US-Casual-K")
    st.caption("🎙️ Available voices: https://cloud.google.com/text-to-speech/docs/list-voices-and-types")
    st.caption("Default: en-US-Casual-K for English and hi-IN-Chirp3-HD-Achird for Hinglish")
with col2:
    speaking_rate = st.slider("🚀 Speaking Rate", 0.5, 2.0, 0.95)
    use_rate = st.checkbox("🗣️ Apply Speaking Rate", value=True)
    pitch = st.slider("🎚️ Pitch", -20.0, 20.0, -2.0)
    use_pitch = st.checkbox("🎵 Apply Pitch", value=True)
    max_bytes = st.slider("🧩 Max Bytes per Chunk", 1000, 6000, 4400)

prompt_override = st.text_area("✍️ Optional: Override Gemini Prompt (use {content})", "", height=150)

# === Step 1: Generate Script ===
if uploaded_file and st.button("🧠 Generate Teaching Script"):
    suffix = uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("🔍 Extracting text and generating script..."):
        raw_text = extract_text(tmp_path)
        if not raw_text:
            st.error("❌ No text extracted.")
        else:
            cleaned = clean_text(raw_text)
            script = generate_teaching_script(cleaned, language_mode, prompt_override)
            if not script:
                st.error("❌ Script generation failed.")
            else:
                st.session_state.generated_script = script
                st.success("✅ Script generated successfully!")

# === Step 2: Display & Edit Script ===
if "generated_script" in st.session_state:
    edited_script = st.text_area("📄 Edit Teaching Script (before audio)", st.session_state.generated_script, height=350)
    st.session_state.edited_script = edited_script

# === Step 3: Generate Audio ===
if "edited_script" in st.session_state and st.button("🔊 Generate Audiobook"):
    with st.spinner("🎧 Synthesizing audio..."):
        audio_path = generate_audio_chunks(
            st.session_state.edited_script,
            voice_name, language_code, speaking_rate, pitch, max_bytes,
            use_rate, use_pitch
        )
        if audio_path:
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
                st.download_button("⬇️ Download Audiobook", f, file_name="audiobook.mp3")
        else:
            st.error("❌ Audio generation failed.")
