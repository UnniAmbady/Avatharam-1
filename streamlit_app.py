# ver-12.2 (fluid mobile layout + fixed avatar + static preview)
# - iPhone-friendly flex rows: Start+hint and Test buttons stay side-by-side.
# - Buttons use intrinsic width (no stretching).
# - Avatar selector removed; fixed to June_HR_public.
# - Static preview shows before Start; replaced by live viewer after Start.
# - Transcript preloaded; mic labels Record/Stop.

import json
import os
import time
import tempfile
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# --- Optional STT backend ---
try:
    from faster_whisper import WhisperModel
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

# Simple text (prevents cropping seen with st.title on some PCs)
st.text("AI Avatar Chat")

# Global CSS — includes mobile tweaks so columns don't stack on phones
st.markdown("""
<style>
  .block-container { padding-top:.6rem; padding-bottom:1rem; }
  iframe { border:none; border-radius:16px; }
  .rowbtn .stButton>button { height:40px; font-size:.95rem; border-radius:12px; }
  .hint { font-size:.92rem; opacity:.85; }

  /* ---- MOBILE FLUID GRID TWEAKS ---- */
  /* Keep horizontal blocks inline on small screens & use intrinsic widths. */
  @media (max-width: 640px) {
    div[data-testid="stHorizontalBlock"] { gap: 8px !important; }
    div[data-testid="column"] {
      flex: 0 0 auto !important;
      width: auto !important;
    }
    .stButton button { width: auto !important; }
  }

  /* Static preview wrapper to mimic live player frame */
  .preview-frame {
    background:#0f0f10;
    border-radius:18px;
    padding:12px 12px 24px 12px;
  }
  .preview-title {
    text-align:center;
    color:#e6e6e6;
    font-weight:600;
    margin:6px 0 10px 0;
  }
  .preview-img-wrap {
    display:flex; align-items:center; justify-content:center;
    overflow:hidden; border-radius:16px; background:#000;
  }
  .preview-img { width:100%; height:auto; object-fit:contain; display:block; }
  .preview-foot { text-align:center; color:#cfcfcf; font-size:.85rem; margin-top:8px; }
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
    if len(ss.debug_buf) > 1000:
        ss.debug_buf[:] = ss.debug_buf[-1000:]

# ------------- HTTP helpers --------------
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

# ---------- Fixed avatar: June_HR_public ----------
PREVIEW_URL = "https://files2.heygen.ai/avatar/v3/74447a27859a456c955e01f21ef18216_45620/preview_talk_1.webp"
FIXED_AVATAR = {
    "avatar_id": "June_HR_public",
    "default_voice": "68dedac41a9f46a6a4271a95c733823c",
    "pose_name": "June HR",
    "is_public": True,
    "normal_preview": PREVIEW_URL,
}
fixed_avatar_id = FIXED_AVATAR["avatar_id"]
fixed_voice_id  = FIXED_AVATAR["default_voice"]
fixed_pose_name = FIXED_AVATAR["pose_name"]

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

# -------------- Controls row (fluid) --------------
st.write("")
row = st.container()
with row:
    c1, c2 = st.columns([1, 5], gap="small")
    with c1:
        # Compact Start button; intrinsic width even on phones
        if st.button("Start", key="start_btn", type="primary"):
            if ss.session_id and ss.session_token:
                stop_session(ss.session_id, ss.session_token); time.sleep(0.2)
            debug("Step 1: streaming.new")
            payload = new_session(fixed_avatar_id, fixed_voice_id)
            sid, offer_sdp, rtc_config = payload["session_id"], payload["offer_sdp"], payload["rtc_config"]
            debug("Step 2: streaming.create_token")
            tok = create_session_token(sid)
            debug("Step 3: sleep 1.0s before viewer")
            time.sleep(1.0)
            ss.session_id, ss.session_token = sid, tok
            ss.offer_sdp, ss.rtc_config = offer_sdp, rtc_config
            debug(f"[ready] session_id={sid[:8]}…")
    with c2:
        # Left-aligned small hint (ASCII arrows)
        st.markdown('<div class="hint">&lt;-&lt;-Click to Start</div>', unsafe_allow_html=True)

# ----------- Viewer / Preview area -----------
viewer_path = Path(__file__).parent / "viewer.html"
viewer_area = st.container()

if ss.session_id and ss.session_token and ss.offer_sdp and viewer_path.exists():
    # Render LIVE viewer (replaces the static preview)
    html = (
        viewer_path.read_text(encoding="utf-8")
        .replace("__SESSION_TOKEN__", ss.session_token)
        .replace("__AVATAR_NAME__", fixed_pose_name)
        .replace("__SESSION_ID__", ss.session_id)
        .replace("__OFFER_SDP__", json.dumps(ss.offer_sdp)[1:-1])  # raw newlines
        .replace("__RTC_CONFIG__", json.dumps(ss.rtc_config or {}))
    )
    with viewer_area:
        components.html(html, height=340, scrolling=False)
else:
    # Static preview frame (neat placeholder before Start)
    with viewer_area:
        st.markdown(
            f"""
<div class="preview-frame">
  <div class="preview-title">Avatar: {fixed_pose_name}</div>
  <div class="preview-img-wrap">
    <img class="preview-img" src="{PREVIEW_URL}" alt="Avatar preview">
  </div>
  <div class="preview-foot">Click Start to open a session and load the viewer.</div>
</div>
""",
            unsafe_allow_html=True,
        )

# =================== Voice Recorder (mic_recorder) ===================
st.write("---")
st.subheader("Voice")
st.caption('Click **Record** to capture audio and **Stop** when done. Transcription runs after stopping.')

wav_bytes: Optional[bytes] = None
if not _HAS_MIC:
    st.warning("`streamlit-mic-recorder` is not installed.")
else:
    audio = mic_recorder(
        start_prompt="Record",   # renamed
        stop_prompt="Stop",
        just_once=False,
        use_container_width=False,
        key="mic_recorder",
        format="wav",
    )

    if audio is None:
        debug("[mic] waiting for recording…")
    else:
        if isinstance(audio, dict) and "bytes" in audio:
            wav_bytes = audio["bytes"]
            debug(f"[mic] received {len(wav_bytes)} bytes")
        elif isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)
            debug(f"[mic] received {len(wav_bytes)} bytes (raw)")
        else:
            debug(f"[mic] unexpected payload: {type(audio)}")

# ---- Audio playback (ABOVE transcript)
if wav_bytes:
    st.audio(wav_bytes, format="audio/wav", autoplay=False)

    # Transcribe (fast-path with faster-whisper, else stub)
    text = ""
    if _HAS_FWHISPER:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes); tmp.flush(); tmp_path = tmp.name
            with st.spinner("Transcribing…"):
                model = WhisperModel("base", compute_type="int8")
                segments, info = model.transcribe(tmp_path, language="en", vad_filter=True)
                text = " ".join([seg.text for seg in segments]).strip()
        except Exception as e:
            debug(f"[whisper] {e}")
        finally:
            try: os.remove(tmp_path)
            except Exception: pass
    if not text:
        text = "Thanks! (audio captured)"

    ss.last_text = text
    debug(f"[voice→text] {text if text else '(empty)'}")

# ---- Transcript box (editable) with preload instructions
DEFAULT_TRANSCRIPT_HINT = (
    'Hello, ok to record the message press "Record" and start speaking. '
    'When the sentence is completed press the "Stop" button.'
)
if not (ss.get("last_text") or "").strip():
    ss.last_text = DEFAULT_TRANSCRIPT_HINT

st.subheader("Transcript")
ss.last_text = st.text_area(" ", value=ss.last_text, height=140, label_visibility="collapsed")

# ============ Actions (fluid two small buttons) ============
st.write("")
act1, act2 = st.columns([1, 1], gap="small")
with act1:
    if st.button("Test-1"):  # intrinsic width
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        else:
            send_echo(ss.session_id, ss.session_token, "Hello. Welcome to the test demonstration.")
with act2:
    if st.button("Test-2 (Send transcript)"):  # intrinsic width
        if not (ss.session_id and ss.session_token and ss.offer_sdp):
            st.warning("Start a session first.")
        elif not (ss.last_text or "").strip():
            st.warning("Transcript is empty.")
        else:
            send_echo(ss.session_id, ss.session_token, ss.last_text.strip())

# -------------- Debug box --------------
st.text_area("Debug", value="\n".join(ss.debug_buf), height=220, disabled=True)
