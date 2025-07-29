# Author: Bilal Saifi
# Version: 8.0 - Added Separate Pitch/Rate Controls for Teacher & Student
# To execute: streamlit run audio-book.py

import os
import re
import html
import tempfile
import json
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
from google.cloud import texttospeech, aiplatform
from vertexai.preview.generative_models import GenerativeModel
import streamlit as st
import datetime
from google.cloud import vision
from google.oauth2 import service_account

logo_url = "https://nostmbdijzudpxxqmcxc.supabase.co/storage/v1/object/public/profile//logo.png"

# === Google Cloud Config ===
#gcp_credentials = json.loads(st.secrets["gcp_service_account"])
gcp_credentials = st.secrets["gcp_service_account"]
credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
project_id = gcp_credentials["project_id"]
location = "us-central1"
aiplatform.init(project=project_id, location=location, credentials=credentials)

LOG_FILE = "token_usage_log.json"
# === Token Usage Logger ===
token_logs = []

def log_tokens(task, text):
    token_count = len(text.split())
    token_logs.append((task, token_count))
    append_token_log(task, token_count)  # persist it


def load_token_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_token_log(log_data):
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)

def append_token_log(task, token_count):
    log_data = load_token_log()
    log_data.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "task": task,
        "tokens": token_count
    })
    save_token_log(log_data)

# === Utility Functions ===
def extract_text(path):
    try:
        client = vision.ImageAnnotatorClient(credentials=credentials)
        if path.endswith(".pdf"):
            pages = convert_from_path(path)
            text = ""
            for i, page in enumerate(pages):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_file:
                    page.save(img_file.name, format="PNG")
                    with open(img_file.name, "rb") as f:
                        content = f.read()
                    image = vision.Image(content=content)
                    response = client.text_detection(image=image)
                    page_text = response.full_text_annotation.text
                    if len(page_text.strip()) > 20:
                        text += f"\n[Page {i+1}]\n{page_text}"
            log_tokens("Vision OCR", text)
            return text
        elif path.endswith((".jpg", ".png", ".jpeg")):
            with open(path, "rb") as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            text = response.full_text_annotation.text
            log_tokens("Vision OCR", text)
            return text
        elif path.endswith(".txt"):
            with open(path, "r") as f:
                text = f.read()
            log_tokens("Text File Read", text)
            return text
    except Exception as e:
        st.error(f"❌ OCR failed: {e}")
    return ""

def clean_text(text):
    text = re.sub(r"\[Page \d+\]", "", text)
    text = re.sub(r"[_*~`!\"]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def sanitize_ssml(text):
    text = html.unescape(text.replace("&", "&amp;"))
    text = re.sub(r'<(\w+)(\s[^>]*)?>', r'<\1>', text)
    if not text.strip().startswith("<speak>"):
        text = f"<speak>{text}</speak>"
    return text

def generate_teaching_script(raw_text, language_mode, prompt_override):
    model = GenerativeModel("gemini-2.5-pro")
    if prompt_override:
        prompt = f"{prompt_override.strip()}\n\nContent:\n{raw_text}"
    elif language_mode == "hinglish":
        prompt = f"""
        Aap ek friendly Indian teacher hain jo JEE Mains aur Advanced ki preparation mein duniya ke sabse best teacher hain. Niche diye gaye content ko students ke liye simple Hinglish mein explain kijiye — Hindi aur English ka mix use karke, jaise hum India mein naturally bolte hain
        Jargon avoid kijiye, lekin technical terms English mein hi rakhiye
        Tone bilkul friendly aur classroom jaise ho
        Wrap Hindi words or phrases in the script using <lang xml:lang="hi-IN">...</lang> inside a <speak> block and Make sure to use hindi font text to write all the hindi words and phrases, to ensure accurate pronunciation when read by Google Text-to-Speech
        write all the words which are to be spoken in hindi in hindi font text so that the pronounciation could be made better
        Short sentences aur easy examples use kijiye
        Baat karte waqt chhoti chhoti pauses lijiye, jaise ek natural teacher leta hai
        Indian pronunciation style ka dhyan rakhiye
        Output sirf plain text mein ho. Koi formatting characters (jaise asterisks *, underscores _, hashtags #, backticks `) ya emojis use na ho.
        Markdown formatting completely avoid karein
        Just plain text jo naturally audio mein bolne jaise lage — clear, spoken explanation without any special characters
        

        Content:
        {raw_text}
        """
    else:
        prompt = f"""
        You are an audiobook narrator and an expert Indian teacher who helps students prepare for JEE and NEET exams.
        Your job is to summarize and explain the following educational content in an easy, short, and student-friendly script.
        The tone should be clear, warm, and friendly, just like a real Indian teacher speaking to students.
        Make the explanation sound natural, like spoken English in India — use expressions and rhythm we commonly use in classrooms.
        Speak with a style that suits Indian pronunciation and pacing.
        Use short sentences, simple words, and engaging language that students can easily understand.
        Include small, natural pauses where a real teacher would pause while speaking.
        Keep the delivery conversational, smooth, and comfortable — as if you’re talking directly to a student in a coaching class.
        ❌ Do not use any special characters such as asterisks *, underscores _, hashtags #, backticks `, or emojis.
        ❌ Avoid all markdown or formatting symbols.
        ✅ Output should be plain text only, exactly how a teacher would speak aloud in an audio recording.
        

        Content:
        {raw_text}
        """
    try:
        response = model.generate_content(prompt)
        log_tokens("Gemini: Teaching Script", response.text)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return ""

def generate_conversation_script(raw_text, language_mode, prompt_override):
    model = GenerativeModel("gemini-2.5-pro-preview-05-06")
    if prompt_override:
        prompt = f"{prompt_override.strip()}\n\nContent:\n{raw_text}"
    elif language_mode == "hinglish":
        prompt = f"""
        Simulate a natural conversation between a friendly Indian teacher and a curious student in Hinglish (mix of Hindi and English).
        The teacher should explain the topic clearly, and the student should occasionally ask doubts in a natural way.
        The teacher replies politely and also asks short questions in between to check the student’s understanding — jaise ek achha teacher karta hai.
        
        ✅ Tone should be very natural and classroom-like, just like how we talk in India.
        ✅ Use short, simple sentences and clear examples.
        ✅ Hindi aur English ka mix hona chahiye (Hinglish), but technical words English mein hi rakho.
        ✅ Teacher ka bolne ka style Indian hona chahiye — jaise real classroom mein bolte hain.
        ✅ Teacher thodi thodi pauses le — jaise naturally bolte time hota hai.
        ✅ Student ke questions bhi genuine aur conversational lagne chahiye.
        ✅ Keep the flow smooth, engaging, and real.
        write all the words which are to be spoken in hindi in hindi font text so that the pronounciation could be made better
        Wrap Hindi words or phrases in the script using <lang xml:lang="hi-IN">...</lang> inside a <speak> block and Make sure to use hindi font text to write all the hindi words and phrases, to ensure accurate pronunciation when read by Google Text-to-Speech
        When generating the script, provide Hindi words in Romanized form (Hinglish) using phonetic spellings that closely match the intended pronunciation. Also wrap the full output in <speak>...</speak> tags to support future SSML tweaks.
        ❌ Strictly avoid any special characters like asterisks *, underscores _, hashtags #, backticks `, or emojis.
        ❌ No markdown formatting at all.
        ✅ Just plain text, written exactly how it would be spoken out loud in a natural Indian conversation.

        Format:
        TEACHER: explanation
        STUDENT: a natural question
        TEACHER: reply
        TEACHER: ask a short question to student
        STUDENT: gives a simple answer

        - Make it sound natural like we speak in India and like teacher
        - Focus more of indian pronounciation style
        - The tone should be natural and clear for students.
        - Make it sound natural like we speak in India and like teacher
        - give small pauses like a natural teacher
        - Focus more of indian pronounciation style
        - Do not use any special characters like asterisks *, markdown formatting, or emojis.
        - Just plain text that sounds like natural speech.
        
        Content:
        {raw_text}
        """
    else:
        prompt = f"""
            Simulate a natural classroom-style conversation between a friendly Indian teacher and a curious student — in English.
            The teacher should explain the topic clearly in a way students preparing for JEE or NEET would easily understand.
            The student should occasionally ask relevant questions, and the teacher should respond politely and also check understanding by asking small follow-up questions.
            
            ✅ The conversation should feel natural, clear, and friendly — like a real Indian teacher explaining something to a student.
            ✅ Use a conversational tone, common Indian speech rhythm, and short pauses like we naturally speak in a classroom.
            ✅ Keep the language simple, with short sentences and real-life analogies or easy-to-grasp examples.
            ✅ Focus on Indian pronunciation style and how teachers explain concepts during verbal sessions.
            
            ❌ Do not use any special characters, such as asterisks *, underscores _, hashtags #, backticks `, or emojis.
            ❌ No markdown formatting or stylized text.
            ✅ Only produce plain text, written exactly like how it would be spoken aloud in an audio recording.
            
            🔄 Use this format exactly:

                TEACHER: (your explanation)
                STUDENT: (a natural question)
                TEACHER: (reply)
                TEACHER: (asks a short question to student)
                STUDENT: (gives a simple answer)

       

        Content:
        {raw_text}
        """
    try:
        response = model.generate_content(prompt)
        log_tokens("Gemini: Conversation Script", response.text)
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

def log_tts_tokens(label, chunks):
    for chunk in chunks:
        token_count = len(chunk.split())
        token_logs.append((f"TTS: {label}", token_count))
        append_token_log(f"TTS: {label}", token_count)

def synthesize_chunks(chunks, voice_name, language_code, speaking_rate, pitch, use_rate, use_pitch):
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    audio_chunks = []
    for chunk in chunks:
        plain_text = re.sub(r"<[^>]+>", "", chunk)
        if not plain_text.strip():
            continue

        log_tts_tokens("Narration", [plain_text])
        input_text = texttospeech.SynthesisInput(text=plain_text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
        config = {"audio_encoding": texttospeech.AudioEncoding.MP3}
        if use_rate: config["speaking_rate"] = speaking_rate
        if use_pitch: config["pitch"] = pitch
        audio_config = texttospeech.AudioConfig(**config)
        try:
            response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
            audio_chunks.append(response.audio_content)
        except Exception as e:
            st.warning(f"Chunk failed: {e}")
    if audio_chunks:
        temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_mp3.name, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)
        return temp_mp3.name
    return None

# --- UPDATED FUNCTION ---
# This function now accepts separate rate/pitch settings for teacher and student.
def generate_conversational_audio(script_lines, teacher_voice, student_voice, language_code, 
                                  teacher_rate, teacher_pitch, use_teacher_rate, use_teacher_pitch,
                                  student_rate, student_pitch, use_student_rate, use_student_pitch):
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    audio_chunks = []
    current_speaker = None
    buffer = []

    def flush():
        nonlocal buffer, current_speaker
        if buffer and current_speaker:
            combined_text = " ".join(buffer).strip()
            if not combined_text:
                buffer = []
                return
            
            voice_name = teacher_voice if current_speaker == "teacher" else student_voice
            plain_text = re.sub(r"<[^>]+>", "", combined_text)
            log_tts_tokens(current_speaker.capitalize(), [plain_text])
            
            input_text = texttospeech.SynthesisInput(text=plain_text)
            voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
            
            # --- CORE LOGIC CHANGE ---
            # Dynamically set pitch and rate based on the current speaker.
            config = {"audio_encoding": texttospeech.AudioEncoding.MP3}
            if current_speaker == "teacher":
                if use_teacher_rate: config["speaking_rate"] = teacher_rate
                if use_teacher_pitch: config["pitch"] = teacher_pitch
            elif current_speaker == "student":
                if use_student_rate: config["speaking_rate"] = student_rate
                if use_student_pitch: config["pitch"] = student_pitch
            
            audio_config = texttospeech.AudioConfig(**config)
            # --- END OF CORE LOGIC CHANGE ---
            
            try:
                response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
                audio_chunks.append(response.audio_content)
            except Exception as e:
                st.warning(f"Block failed ({current_speaker}): {e}")
        buffer = []

    for line in script_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(teacher|student)\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            flush()
            current_speaker = match.group(1).lower()
            buffer = [match.group(2).strip()]
        elif buffer: # Append to existing speaker's buffer if it's a continuation line
            buffer.append(line)
    flush() # Flush the last speaker's buffer

    if audio_chunks:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
            for chunk in audio_chunks:
                temp_mp3.write(chunk)
            return temp_mp3.name
    return None

# === Streamlit UI ===
st.set_page_config(page_title="AI Audiobook Generator", layout="wide")
st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="{logo_url}" alt="Logo" width="80" style="margin-right: 15px;">
        <h1 style="margin: 0; font-size: 28px;">
            <a href="https://sparkl.me" target="_blank" style="text-decoration: none; color: inherit;">
                Sparkl Edventure
            </a>
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.title("🎙️ AI-Powered Audiobook Generator")

uploaded_file = st.file_uploader("📂 Upload PDF, Image, or Text File", type=["pdf", "png", "jpg", "jpeg", "txt"])
conversation_mode = st.checkbox("🧠 Enable Conversation Mode")

col1, col2 = st.columns(2)
with col1:
    language_mode = st.selectbox("🗣️ Language Style", ["english", "hinglish"])
    language_code = st.text_input("🌐 TTS Language Code", "en-US")
    if conversation_mode:
        teacher_voice = st.text_input("👨‍🏫 Teacher Voice Name", "en-US-Casual-K")
        student_voice = st.text_input("🧑‍🎓 Student Voice Name", "en-US-Standard-F")
    else:
        voice_name = st.text_input("🎙️ TTS Voice Name", "en-US-Casual-K")

    st.info("Available voices: [Google Cloud TTS Docs](https://cloud.google.com/text-to-speech/docs/voices)")
    st.caption("Defaults (English): Teacher `en-US-Casual-K`, Student `en-US-Standard-F`")
    st.caption("Defaults (Hinglish): Teacher `hi-IN-Neural2-C`, Student `hi-IN-Neural2-A`")

# --- UPDATED UI SECTION ---
# The UI now conditionally shows controls based on conversation_mode.
with col2:
    if conversation_mode:
        st.subheader("👨‍🏫 Teacher Voice Settings")
        teacher_rate = st.slider("🚀 Teacher Speaking Rate", 0.5, 2.0, 0.95, key="teacher_rate")
        use_teacher_rate = st.checkbox("Apply Teacher Rate", value=True, key="use_teacher_rate")
        teacher_pitch = st.slider("🎚️ Teacher Pitch", -20.0, 20.0, -2.0, key="teacher_pitch")
        use_teacher_pitch = st.checkbox("Apply Teacher Pitch", value=True, key="use_teacher_pitch")

        st.subheader("🧑‍🎓 Student Voice Settings")
        student_rate = st.slider("🚀 Student Speaking Rate", 0.5, 2.0, 1.05, key="student_rate")
        use_student_rate = st.checkbox("Apply Student Rate", value=True, key="use_student_rate")
        student_pitch = st.slider("🎚️ Student Pitch", -20.0, 20.0, 0.0, key="student_pitch")
        use_student_pitch = st.checkbox("Apply Student Pitch", value=True, key="use_student_pitch")
    else:
        st.subheader("🎙️ Voice Settings")
        speaking_rate = st.slider("🚀 Speaking Rate", 0.5, 2.0, 0.95)
        use_rate = st.checkbox("🗣️ Apply Speaking Rate", value=True)
        pitch = st.slider("🎚️ Pitch", -20.0, 20.0, -2.0)
        use_pitch = st.checkbox("🎵 Apply Pitch", value=True)
    
    st.subheader("⚙️ Technical Settings")
    max_bytes = st.slider("🧩 Max Bytes per Chunk (for single narrator mode)", 1000, 6000, 4400)

prompt_override = st.text_area("✍️ Optional: Override Gemini Prompt (use {raw_text} to include content)", "", height=150)

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
            st.session_state.raw_text_for_prompt = cleaned # Store for optional override
            
            # Prepare the prompt based on override
            prompt_to_use = prompt_override.format(raw_text=cleaned) if prompt_override else ""

            if conversation_mode:
                script = generate_conversation_script(cleaned, language_mode, prompt_to_use)
            else:
                script = generate_teaching_script(cleaned, language_mode, prompt_to_use)
                
            if not script:
                st.error("❌ Script generation failed.")
            else:
                st.session_state.generated_script = script
                st.success("✅ Script generated successfully!")
    
    os.remove(tmp_path)


if "generated_script" in st.session_state:
    edited_script = st.text_area("📄 Edit Script (before audio)", st.session_state.generated_script, height=350)
    st.session_state.edited_script = edited_script

# --- UPDATED AUDIO GENERATION LOGIC ---
# Removed the `generate_audio` wrapper and call specific functions directly.
if "edited_script" in st.session_state and st.button("🔊 Generate Audiobook"):
    with st.spinner("🎧 Synthesizing audio... This may take a moment."):
        script = st.session_state.edited_script
        audio_path = None 
        
        if conversation_mode:
            audio_path = generate_conversational_audio(
                script_lines=script.splitlines(),
                teacher_voice=teacher_voice,
                student_voice=student_voice,
                language_code=language_code,
                teacher_rate=teacher_rate,
                teacher_pitch=teacher_pitch,
                use_teacher_rate=use_teacher_rate,
                use_teacher_pitch=use_teacher_pitch,
                student_rate=student_rate,
                student_pitch=student_pitch,
                use_student_rate=use_student_rate,
                use_student_pitch=use_student_pitch
            )
        else: # Standard mode
            chunks = split_by_bytes(script, max_bytes=max_bytes)
            audio_path = synthesize_chunks(
                chunks=chunks,
                voice_name=voice_name,
                language_code=language_code,
                speaking_rate=speaking_rate,
                pitch=pitch,
                use_rate=use_rate,
                use_pitch=use_pitch
            )

        if audio_path:
            with open(audio_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
                st.download_button("⬇️ Download Audiobook", f, file_name="audiobook.mp3")
            os.remove(audio_path)
        else:
            st.error("❌ Audio generation failed.")

with st.expander("📊 View Token Usage Logs"):
    log_data = load_token_log()
    if log_data:
        total_tokens = sum(entry["tokens"] for entry in log_data)
        st.markdown(f"**Cumulative Total Tokens:** `{total_tokens}`")

        # Display latest 50 logs
        st.write("---")
        st.write("**Latest 50 Entries:**")
        for entry in reversed(log_data[-50:]):
            st.markdown(f"- `{entry['timestamp']}` | **{entry['task']}**: {entry['tokens']} tokens")
    else:
        st.info("No token logs yet.")
