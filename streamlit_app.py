# streamlit_app.py
# Mobile-first Streamlit app: HeyGen Avatar (Alessandra Casual Look) + Start/Stop mic.
# Behavior: When mic is ON, your speech is transcribed and immediately echoed by the avatar.
# Notes:
# - Reads HeyGen API key from st.secrets["HeyGen"]["heygen_api_key"].
# - (Optional) Reads OpenAI API key from st.secrets["openai"]["api_key"] if you use Whisper.
# - Designed for iPhone/smartphone screens (single-column, big buttons, no debug panes).
# - Minimal external controls: only Start and Stop.
#
# Dependencies (add these to requirements.txt):
#   streamlit==1.38.0
#   streamlit-webrtc==0.47.7
#   requests==2.32.3
#   pydub==0.25.1
#   openai==1.43.0   # optional but recommended for robust STT; you can swap with your preferred STT
#   numpy==1.26.4
#   av==12.2.0
#
# If you prefer to avoid OpenAI Whisper for STT, swap the transcribe_audio_bytes() implementation
# with another STT provider of your choice.

import os
import uuid
import time
import json
import queue
import base64
import threading
from typing import Optional

import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, AudioProcessorBase

# ----------------------------------
# Config & Secrets
# ----------------------------------
st.set_page_config(page_title="Avatar Echo — Alessandra", page_icon="🗣️", layout="centered")

HEYGEN_API_KEY = st.secrets["HeyGen"]["heygen_api_key"]
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")  # optional

AVATAR_ID = "Alessandra_CasualLook_public"
VOICE_ID = "0d3f35185d7c4360b9f03312e0264d59"

# The HeyGen Interactive Avatar (Streaming) iframe base. This is the same viewer used in your stage-2 app.
# If your org uses a different base URL for the viewer, update the value below.
HEYGEN_IFRAME_BASE = "https://labs.heygen.com/streaming"  # common default; adjust if your code uses a different host

# ----------------------------------
# Simple CSS for mobile (big buttons, full-width, minimal chrome)
# ----------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.8rem; padding-bottom: 2rem; }
      .stButton > button { width: 100%; height: 56px; font-size: 1.1rem; border-radius: 12px; }
      .pill { background: #f1f5f9; padding: 6px 12px; border-radius: 999px; font-size: 0.85rem; }
      .caption { color: #6b7280; font-size: 0.8rem; }
      iframe { border: none; border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------
# Utilities
# ----------------------------------

def _ok(resp: requests.Response) -> bool:
    return 200 <= resp.status_code < 300

@st.cache_data(show_spinner=False)
def create_heygen_token(avatar_id: str, voice_id: str) -> dict:
    """Create a short-lived HeyGen streaming token and session.
    Returns dict with fields: {"token": str, "session_id": str, "realtime_endpoint": str}
    """
    url = "https://api.heygen.com/v1/streaming.create_token"
    payload = {"voice_id": voice_id, "avatar_id": avatar_id}
    headers = {"x-api-key": HEYGEN_API_KEY, "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if not _ok(resp):
        raise RuntimeError(f"HeyGen token error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Expected shape: {"data": {"token": "...", "session_id": "...", "realtime_endpoint": "wss://..."}}
    return data.get("data", data)

# Optional: If your stage-2 code posts text to the same session via REST, expose a helper.
# Confirm the exact endpoint your code uses; the line below follows the common pattern.

def speak_text_via_session(session_id: str, text: str) -> None:
    """Send a text input to the existing HeyGen realtime session so the avatar speaks it.
    Adjust the endpoint/path if your stage-2 code uses a slightly different route.
    """
    url = f"https://api.heygen.com/v1/streaming/session/{session_id}/speak"
    headers = {"x-api-key": HEYGEN_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if not _ok(resp):
        st.toast("Avatar speak call failed — check endpoint path in code.")

# ----------------------------------
# Transcription (OpenAI Whisper, optional but recommended)
# ----------------------------------

def transcribe_audio_bytes(raw_pcm: bytes, sample_rate: int) -> Optional[str]:
    """Send audio to Whisper for transcription and return text. PCM 16-bit mono expected.
    Returns None on failure or empty result.
    """
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        import tempfile
        import wave

        client = OpenAI(api_key=OPENAI_API_KEY)
        # Write WAV (16-bit PCM mono)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(raw_pcm)
        with open(wav_path, "rb") as f:
            tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = tr.text.strip() if hasattr(tr, "text") else None
        return text or None
    except Exception as e:
        st.toast(f"STT error: {e}")
        return None

# ----------------------------------
# Audio pipeline (WebRTC → PCM buffer → transcription → avatar speak)
# ----------------------------------

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self._last_ts = time.time()
        self._pcm_queue: "queue.Queue[bytes]" = queue.Queue()
        self.sample_rate = 16000  # we will resample to 16 kHz mono if needed

    def recv_audio(self, frame):
        # frame: av.AudioFrame
        # Convert to 16 kHz mono, 16-bit PCM
        pcm16 = frame.to_ndarray(format="s16")
        # If stereo, take first channel
        if pcm16.ndim == 2 and pcm16.shape[0] > 1:
            pcm16 = pcm16[0:1, :]
        pcm16 = np.squeeze(pcm16)
        # Resample if needed
        in_rate = frame.sample_rate
        if in_rate != self.sample_rate:
            # Simple linear resample; for short chunks it's okay.
            duration = pcm16.shape[0] / in_rate
            new_length = int(duration * self.sample_rate)
            pcm16 = np.interp(
                np.linspace(0, pcm16.shape[0], new_length, endpoint=False),
                np.arange(pcm16.shape[0]),
                pcm16.astype(np.float32),
            ).astype(np.int16)
        self._pcm_queue.put(pcm16.tobytes())
        return frame

    def read_all(self) -> bytes:
        chunks = []
        try:
            while True:
                chunks.append(self._pcm_queue.get_nowait())
        except queue.Empty:
            pass
        return b"".join(chunks)

# Background worker that periodically grabs audio, transcribes, and sends to avatar

def start_worker(stop_event: threading.Event, session_id: str, proc: MicAudioProcessor, interval_sec: float = 1.2):
    while not stop_event.is_set():
        time.sleep(interval_sec)
        raw = proc.read_all()
        if not raw:
            continue
        text = transcribe_audio_bytes(raw, sample_rate=proc.sample_rate)
        if text:
            speak_text_via_session(session_id, text)

# ----------------------------------
# UI — single column, mobile-first
# ----------------------------------

st.markdown("<div class='pill'>Alessandra • Echo mode</div>", unsafe_allow_html=True)

# 1) Create a HeyGen session & token (no button presses; auto on load)
with st.spinner("Preparing avatar…"):
    token_data = create_heygen_token(AVATAR_ID, VOICE_ID)
    session_id = token_data.get("session_id")
    token = token_data.get("token")

# 2) Render the avatar viewer (iframe). No extra controls.
#    The viewer URL format typically accepts token via query string.
iframe_src = f"{HEYGEN_IFRAME_BASE}?token={token}"
st.components.v1.iframe(iframe_src, height=420)

# 3) Mic controls (Start/Stop)
if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None
if "worker_thread" not in st.session_state:
    st.session_state.worker_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "audio_proc" not in st.session_state:
    st.session_state.audio_proc = MicAudioProcessor()

col1, col2 = st.columns(2, gap="small")

with col1:
    start_clicked = st.button("▶️  Start", type="primary")
with col2:
    stop_clicked = st.button("⏹️  Stop")

# Start
if start_clicked:
    # (Re)create stop/event flags
    if st.session_state.worker_thread and st.session_state.worker_thread.is_alive():
        st.session_state.stop_event.set()
        st.session_state.worker_thread.join(timeout=1.0)
    st.session_state.stop_event = threading.Event()

    # Launch WebRTC (audio only)
    st.session_state.webrtc_ctx = webrtc_streamer(
        key=f"mic-{uuid.uuid4()}",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=lambda: st.session_state.audio_proc,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=False,
    )

    # Background worker for STT → avatar speak
    st.session_state.worker_thread = threading.Thread(
        target=start_worker,
        args=(st.session_state.stop_event, session_id, st.session_state.audio_proc, 1.2),
        daemon=True,
    )
    st.session_state.worker_thread.start()
    st.toast("Mic started. Say something — Alessandra will echo it.")

# Stop
if stop_clicked:
    try:
        if st.session_state.stop_event:
            st.session_state.stop_event.set()
        if st.session_state.worker_thread and st.session_state.worker_thread.is_alive():
            st.session_state.worker_thread.join(timeout=1.0)
        st.session_state.worker_thread = None
        st.toast("Mic stopped.")
    finally:
        # Close WebRTC
        if st.session_state.get("webrtc_ctx") is not None:
            st.session_state.webrtc_ctx.destroy()
            st.session_state.webrtc_ctx = None

# Footnote for small screens
st.markdown(
    "<div class='caption'>Tip: For best iPhone results, open in Safari and allow microphone access.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------
# Implementation notes (read me)
# ----------------------------------
# 1) The speak_text_via_session() endpoint path may vary depending on your current stage-2 integration.
#    If your existing code uses a different route (e.g., /v1/streaming.input, /v1/realtime/sessions/{id}/input, etc.),
#    adjust that function accordingly. The rest of the app remains the same.
# 2) If your Alessandra viewer expects additional query params (pose, voice_id, etc.),
#    append them to iframe_src. Token alone is often sufficient once generated with the desired avatar+voice.
# 3) If you prefer a push-to-talk UX on mobile, replace Start with a toggle or hold-to-talk using a small custom component.
# 4) If you want to remove any potential Streamlit status messages, set server.headless = true and client.showErrorDetails = false in config.
