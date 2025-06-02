# Author: Bilal Saifi
# Version: 2.5 - Streamlit Cloud Compatible with Vision OCR
# Notes: for more audios https://cloud.google.com/text-to-speech/docs/list-voices-and-types
# To execute this code run: streamlit run audio-book.py

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

# Fix for vision import
from google.cloud import vision
from google.oauth2 import service_account

# === Google Cloud Config (read from secrets.toml on Streamlit Cloud) ===
gcp_credentials = json.loads(st.secrets["gcp_service_account"])
credentials = service_account.Credentials.from_service_account_info(gcp_credentials)

project_id = gcp_credentials["project_id"]
location = "us-central1"
aiplatform.init(project=project_id, location=location, credentials=credentials)

# === Utility Functions ===
def extract_text(path):
    try:
        client = vision.ImageAnnotatorClient(credentials=credentials)
        content = None

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
            return text

        elif path.endswith((".jpg", ".png", ".jpeg")):
            with open(path, "rb") as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            return response.full_text_annotation.text

        elif path.endswith(".txt"):
            with open(path, "r") as f:
                return f.read()

    except Exception as e:
        st.error(f"❌ OCR failed: {e}")
    return ""

# (rest of the code remains unchanged)
