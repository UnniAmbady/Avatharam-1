# ver-11.1
# HeyGen avatar + mic_recorder + (optional) faster-whisper
# Changes vs ver-11:
# - Viewer embed height reduced ~50% (fits on one screen).
# - Mic recorder with audio playback shown below buttons and above transcript box.
# - Keeps transcript in st.session_state.last_text.
# - Adds Test-2 (send transcript to avatar). Removes Clear/Reset.

import json
import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components

# --- Optional STT backends ---
try:
    from faster_whisper import WhisperModel  # local inference
    _HAS_FWHISPER = True
except Exception:
    WhisperModel = None  # type: ignore
    _HAS_FWHISPER = False

try:
    from streamlit_mic_recorder import mic_recorder
    _HAS_MIC = True
except Exception:
    mic_recorder = None  # type: ignore
    _HAS_MIC = False

# ---------------- Page ----------------
st.set_page_config(page_title="AI Avatar Demo", layout="centered")
st.title("AI Avatar Demo")

st.markdown("""
<style>
  .block-container { padding-top:.6rem; padding-bottom:1rem; }
  iframe { border:none; border-radius:16px; }
  .rowbtn .stButton>button { height:40px; font-size:.95rem; border-radius:12px; }
  .hint { font-size:.9rem; opacity:.7; }
</style>
""", unsafe_allow_html=True)

# --------------- Secrets ---------------
def _get(s: dict, *keys, default=None):
    cur = s
    try:
        for k in keys: cur = cur[k]
        return cur
    except Exception:
        return default

SECRETS = st.secrets if "secrets" in dir(st) else {}
HEYGEN_API_KEY = (_get(SECRETS, "HeyGen", "heygen_api_key")
                  or _get(SECRETS, "heygen", "heygen_api_key")
                  or os.getenv("HEYGEN_API_KEY"))
if not HEYGEN_API_KEY:
    st.error("Missing HeyGen API key in `.streamlit/secrets.toml`.\n\n[HeyGen]\nheygen_api_key = \"…\"")
    st.stop()

# --------------- Endpoints --------------
BASE = "https://api.heygen.com/v1"
API_LIST_AVATARS = f"{BASE}/streaming/avatar.list"     # GET (x-api-key)
API_STREAM_NEW   = f"{BASE}/streaming.new"             # POST (x-api-key)
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"    # POST (x-api-key)
API_STREAM_TASK  = f"{BASE}/streaming.task"            # POST (Bearer)
API_STREAM_STOP  = f"{BASE}/streaming.stop"            # POST (Bearer)

HEADERS_XAPI = {"accept":"application/json","x-api-key":HEYGEN_API_KEY,"Content-Type":"application/json"}
def headers_bearer(tok: str):
    return {"accept":"application/json","Authorization":f"Bearer {tok}","Content-Type":"application/json"}

# --------- Debug buffer ----------
ss = st.session_state
ss.setdefault("debug_buf", [])
def debug(msg: str):
    ss.debug_buf.append(str(msg))
    if len(ss.debug_buf) > 1000: ss.debug_buf[:] = ss.debug_buf[-1000:]

# ------------- HTTP helpers --------------
def _get(url, params=None):
    r = requests.get(url, headers=HEADERS_XAPI, params=params, timeout=45)
    raw = r.text
    try: body = r.json()
    except Exception: body = {"_raw": raw}
    debug(f"[GET] {url} -> {r.status_code}")
    if r.status_code >= 400: debug(raw); r.raise_for_status()
    return r.status_code, body, raw

def _post_xapi(url, payload=None):
    r = requests.post(url, headers=HEADERS_XAPI, data=json.dumps(payload or {}), timeout=60)
    raw = r.text
    try: body = r.json()
    except Exception: body = {"_raw": raw}
    debug(f"[POST x-api] {url} -> {r.status_code}")
    if r.status_code >= 400: debug(raw); r.raise_for_status()
    return r.status_code, body, raw

def _post_bearer(url, token, payload=None):
    r = requests.post(url, headers=headers_bearer(token), data=json.dumps(payload or {}), timeout=60)
    raw = r.text
    try: body = r.json()
    except Exception: body = {"_raw": raw}
    debug(f"[POST bearer] {url} -> {r.status_code}")
    if r.status_code >= 400: debug(raw); r.raise_for_status()
    return r.status_code, body, raw

# --------- Avatars (ACTIVE only) ---------
@st.cache_data(ttl=300)
def fetch_interactive_avatars():
    _, body, _ = _get(API_LIST_AVATARS)
    items = []
    for a in (body.get("data") or []):
        if isinstance(a, dict) and a.get("status") == "ACTIVE":
            items.append({
                "label": a.get("pose_name") or a.get("avatar_id"),
                "avatar_id": a.get("avatar_id"),
                "default_voice": a.get("default_voice"),
            })
    seen, out = set(), []
    for it in items:
        aid = it.get("avatar_id")
        if aid and aid not in seen:
            seen.add(aid); out.append(it)
    return out

avatars = fetch_interactive_avatars()
if not avatars:
    st.error("No ACTIVE interactive avatars returned by HeyGen.")
    st.stop()

# Default to Alessandra if present
default_idx = 0
for i, a in enumerate(avatars):
    if a["avatar_id"] == "Alessandra_CasualLook_public":
        default_idx = i; break

choice = st.selectbox("Choose an avatar", [a["label"] for a in avatars], index=default_idx)
selected = next(a for a in avatars if a["label"] == choice)

# ------------- Session helpers -------------
def new_session(avatar_id: str, voice_id: Optional[str] = None):
    payload = {"avatar_id": avatar_id}
    if voice_id: payload["voice_id"] = voice_id
    _, body, _ = _post_xapi(API_STREAM_NEW, payload)
    data = body.get("data") or {}
    sid = data.get("session_id")
    offer_sdp = (data.get("offer") or data.get("sdp") or {}).get("sdp")
    ice2 = data.get("ice_servers2"); ice1 = data.get("ice_servers")
    if isinstance(ice2, list) and ice2: rtc_config = {"iceServers": ice2}
    elif isinstance(ice1, list) and ice1: rtc_config = {"iceServers": ice1}
    else: rtc_config = {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
    if not sid or not offer_sdp: raise RuntimeError(f"Missing session_id or offer in response: {body}")
    return {"session_id": sid, "offer_sdp": offer_sdp, "rtc_config": rtc_config}

def create_session_token(session_id: str) -> str:
    _, body, _ = _post_xapi(API_CREATE_TOKEN, {"session_id": session_id})
    tok = (body.get("data") or {}).get("token") or (body.get("data") or {}).get("access_token")
    if not tok: raise RuntimeError(f"Missing token in response: {body}")
    return tok

def send_echo(session_id: str, session_token: str, text: str):
    debug(f"[echo] {text}")
    _post_bearer(API_STREAM_TASK, session_token, {
        "session_id": session_id,
        "task_type": "repeat",
        "task_mode": "sync",
        "text": text
    })

def stop_session(session_id: str, session_token: str):
    try: _post_bearer(API_STREAM_STOP, session_token, {"session_id": session_id})
    except Exception as e: debug(f"[stop_session] {e}")

# ---------- Streamlit state ----------
ss.setdefault("session_id", None)
ss.setdefault("session_token", None)
ss.setdefault("offer_sdp", None)
ss.setdefault("rtc_config", None)
ss.setdefault("last_text", "")

# -------------- Controls row --------------
st.write("")
c1, c2 = st.columns(2)
with c1:
    if st.button("Start / Restart", use_container_width=True):
        if ss.session_id and ss.session_token:
            stop_session(ss.session_id, ss.session_token); time.sleep(0.2)
        debug("Step 1: streaming.new")
        payload = new_session(selected["avatar_id"], selected.get("default_voice"))
        sid, offer_sdp, rtc_config = payload["session_id"], payload["offer_sdp"], payload["rtc_config"]
        debug("Step 2: streaming.create_token")
        tok = create_session_token(sid)
        debug("Step 3: sleep 1.0s before viewer")
        time.sleep(1.0)
        ss.session_id, ss.session_token = sid, tok
        ss.offer_sdp, ss.rtc_config = offer_sdp, rtc_config
        debug(f"[ready] session_id={sid[:8]}…")
with c2:
    if st.button("Stop", type="secondary", use_container_width=True):
        if ss.session_id and ss.session_token:
            stop_session(ss.session_id, ss.session_token)
        ss.session_id = None; ss.session_token = None
        ss.offer_sdp = None; ss.rtc_config = None
        debug("[stopped] session cleared")

# ----------- Viewer embed (reduced height) -----------
viewer_path = Path(__file__).parent / "viewer.html"
if not viewer_path.exists():
    st.warning("viewer.html not found next to streamlit_app.py.")
else:
    if ss.session_id and ss.session_token and ss.offer_sdp:
        html = (
            viewer_path.read_text(encoding="utf-8")
            .replace("__SESSION_TOKEN__", ss.session_token)
            .replace("__AVATAR_NAME__", selected["label"])
            .replace("__SESSION_ID__", ss.session_id)
            .replace("__OFFER_SDP__", json.dumps(ss.offer_sdp)[1:-1])  # raw newlines
            .replace("__RTC_CONFIG__", json.dumps(ss.rtc_config or {}))
        )
        # height reduced from ~620 to 320 (≈50%)
        components.html(html, height=320, scrolling=False)
    else:
        st.info("Click **Start / Restart** to open a session and load the viewer.")

# =================== Voice Recorder (mic_recorder) ===================
st.write("---")
st.subheader("Voice")
st.caption("Tap **Start** to allow microphone • Speak • Tap **Stop**. The audio bar appears below; transcript is sent to the avatar via Test-2.")

wav_bytes: Optional[bytes] = None
if not _HAS_MIC:
    st.warning("`streamlit-mic-recorder` is not installed.")
else:
    # The mic widget (top-left 'Start' / 'Stop' as in your working app)
    audio = mic_recorder(
        start_prompt="Start",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=False,
        format="wav",
        key=f"mic_{int(time.time())}",  # ensure fresh
    )
    if audio:
        if isinstance(audio, dict) and "bytes" in audio:
            wav_bytes = audio["bytes"]
        elif isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)

# ---- Audio playback (shown ABOVE transcript)
if wav_bytes:
    st.audio(wav_bytes, format="audio/wav", autoplay=False)

    # Transcribe (fast-path with faster-whisper, else stub)
    text = ""
    if _HAS_FWHISPER:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes); tmp.flush(); tmp_path = tmp.name
            model = WhisperModel("base", compute_type="int8")
            segments, info = model.transcribe(tmp_path, language="en", vad_filter=True)
            text = " ".join([seg.text for seg in segments]).strip()
        finally:
            try: os.remove(tmp_path)
            except Exception: pass
    else:
        # Fallback stub text (lets the pipeline work even without Whisper)
        import wave, io as _io
        try:
            with wave.open(_io.BytesIO(wav_bytes), "rb") as w:
                frames = w.getnframes(); rate = w.getframerate()
                secs = frames / float(rate or 16000)
            text = f"I heard you for about {secs:.1f} seconds."
        except Exception:
            text = "Thanks! (audio captured)"

    ss.last_text = text
    debug(f"[voice→text] {text if text else '(empty)'}")

# ---- Transcript box (editable)
st.subheader("Transcript")
ss.last_text = st.text_area(" ", value=ss.last_text, height=140, label_visibility="collapsed")

# ============ Quick actions ============
st.write("")
col1, col2 = st.columns(2, gap="small")
with col1:
    if st.button("Test-1", use_container_width=True):
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        else:
            send_echo(ss.session_id, ss.session_token, "Hello. Welcome to the test demonstration.")
with col2:
    if st.button("Test-2 (Send transcript)", use_container_width=True):
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        elif not (ss.last_text or "").strip():
            st.warning("Transcript is empty.")
        else:
            send_echo(ss.session_id, ss.session_token, ss.last_text.strip())

# -------------- Debug box --------------
st.text_area("Debug", value="\n".join(ss.debug_buf), height=220, disabled=True)
