#ver-10.2
# HeyGen — Stage.2.Ver.7 base + Voice Echo (Start/Stop) + Debug box
# Sequence: streaming.new -> streaming.create_token -> sleep(1s) -> viewer.start
# UI aligned with your working app (title, Start/Restart, Stop, Test-1).

import json
import time
import os
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# Optional voice (Whisper) + mic
try:
    import numpy as np
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    _HAS_WEBRTC = True
except Exception:
    _HAS_WEBRTC = False

# ---------------- Page ----------------
st.set_page_config(page_title="AI Avatar Demo", layout="centered")
st.title("AI Avatar Demo")

st.markdown("""
<style>
  .block-container { padding-top:.5rem; }
  .ctrl-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .ctrl-row .stButton > button { height:40px; padding:0 12px; font-size:.9rem; border-radius:10px; }
  iframe { border:none; border-radius:16px; }
  /* hide internal start of streamlit-webrtc to avoid the flashing red button */
  div.st-webrtc > div:has(button) { display:none !important; }
</style>
""", unsafe_allow_html=True)

# --------------- Secrets ---------------
def _get(s: dict, *keys, default=None):
    cur = s
    try:
        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return default

SECRETS = st.secrets if "secrets" in dir(st) else {}
HEYGEN_API_KEY = _get(SECRETS, "HeyGen", "heygen_api_key") or _get(SECRETS, "heygen", "heygen_api_key") or os.getenv("HEYGEN_API_KEY")
if not HEYGEN_API_KEY:
    st.error("Missing HeyGen API key in `.streamlit/secrets.toml`.\n\n[HeyGen]\nheygen_api_key = \"…\"")
    st.stop()

OPENAI_API_KEY = _get(SECRETS, "openai", "secret_key") or _get(SECRETS, "openai", "api_key") or os.getenv("OPENAI_API_KEY")

# --------------- Endpoints --------------
BASE = "https://api.heygen.com/v1"
API_LIST_AVATARS = f"{BASE}/streaming/avatar.list"     # GET (x-api-key)
API_STREAM_NEW   = f"{BASE}/streaming.new"             # POST (x-api-key)
API_CREATE_TOKEN = f"{BASE}/streaming.create_token"    # POST (x-api-key)
API_STREAM_TASK  = f"{BASE}/streaming.task"            # POST (Bearer)
API_STREAM_STOP  = f"{BASE}/streaming.stop"            # POST (Bearer)

HEADERS_XAPI = {
    "accept": "application/json",
    "x-api-key": HEYGEN_API_KEY,
    "Content-Type": "application/json",
}
def headers_bearer(token: str):
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

# --------- Debug buffer (UI text area) ----------
ss = st.session_state
ss.setdefault("debug_buf", [])
def debug(msg: str):
    ss.debug_buf.append(str(msg))
    # Trim to keep the box responsive
    if len(ss.debug_buf) > 500:
        ss.debug_buf[:] = ss.debug_buf[-500:]

# ------------- HTTP helpers --------------
def _get(url, params=None):
    r = requests.get(url, headers=HEADERS_XAPI, params=params, timeout=45)
    raw = r.text
    try:
        body = r.json()
    except Exception:
        body = {"_raw": raw}
    debug(f"[GET] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(raw)
        r.raise_for_status()
    return r.status_code, body, raw

def _post_xapi(url, payload=None):
    r = requests.post(url, headers=HEADERS_XAPI, data=json.dumps(payload or {}), timeout=60)
    raw = r.text
    try:
        body = r.json()
    except Exception:
        body = {"_raw": raw}
    debug(f"[POST x-api] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(raw)
        r.raise_for_status()
    return r.status_code, body, raw

def _post_bearer(url, token, payload=None):
    r = requests.post(url, headers=headers_bearer(token), data=json.dumps(payload or {}), timeout=60)
    raw = r.text
    try:
        body = r.json()
    except Exception:
        body = {"_raw": raw}
    debug(f"[POST bearer] {url} -> {r.status_code}")
    if r.status_code >= 400:
        debug(raw)
        r.raise_for_status()
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
    # dedupe
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
    if voice_id:
        payload["voice_id"] = voice_id
    _, body, _ = _post_xapi(API_STREAM_NEW, payload)
    data = body.get("data") or {}

    sid = data.get("session_id")
    offer_sdp = (data.get("offer") or data.get("sdp") or {}).get("sdp")
    ice2 = data.get("ice_servers2")
    ice1 = data.get("ice_servers")
    if isinstance(ice2, list) and ice2:
        rtc_config = {"iceServers": ice2}
    elif isinstance(ice1, list) and ice1:
        rtc_config = {"iceServers": ice1}
    else:
        rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    if not sid or not offer_sdp:
        raise RuntimeError(f"Missing session_id or offer in response: {body}")

    return {"session_id": sid, "offer_sdp": offer_sdp, "rtc_config": rtc_config}

def create_session_token(session_id: str) -> str:
    _, body, _ = _post_xapi(API_CREATE_TOKEN, {"session_id": session_id})
    tok = (body.get("data") or {}).get("token") or (body.get("data") or {}).get("access_token")
    if not tok:
        raise RuntimeError(f"Missing token in response: {body}")
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
    try:
        _post_bearer(API_STREAM_STOP, session_token, {"session_id": session_id})
    except Exception as e:
        debug(f"[stop_session] {e}")

# ---------- Streamlit state ----------
ss.setdefault("session_id", None)
ss.setdefault("session_token", None)
ss.setdefault("offer_sdp", None)
ss.setdefault("rtc_config", None)

# -------------- Controls row --------------
st.write("")
c1, c2 = st.columns(2)
with c1:
    if st.button("Start / Restart", use_container_width=True):
        if ss.session_id and ss.session_token:
            stop_session(ss.session_id, ss.session_token)
            time.sleep(0.2)

        debug("Step 1: streaming.new")
        payload = new_session(selected["avatar_id"], selected.get("default_voice"))
        sid, offer_sdp, rtc_config = payload["session_id"], payload["offer_sdp"], payload["rtc_config"]

        debug("Step 2: streaming.create_token")
        tok = create_session_token(sid)

        debug("Step 3: sleep 1.0s before viewer")
        time.sleep(1.0)

        ss.session_id = sid
        ss.session_token = tok
        ss.offer_sdp = offer_sdp
        ss.rtc_config = rtc_config
        debug(f"[ready] session_id={sid[:8]}…")

with c2:
    if st.button("Stop", type="secondary", use_container_width=True):
        if ss.session_id and ss.session_token:
            stop_session(ss.session_id, ss.session_token)
        ss.session_id = None
        ss.session_token = None
        ss.offer_sdp = None
        ss.rtc_config = None
        debug("[stopped] session cleared")

# ----------- Viewer embed (exactly as working) -----------
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
        components.html(html, height=620, scrolling=False)
    else:
        st.info("Click **Start / Restart** to open a session and load the viewer.")

# ---------- Buttons row (keep Test-1 identical) ----------
st.write("---")
b1, b2, b3, b4, b5 = st.columns(5)
def _need_session():
    return not (ss.session_id and ss.session_token and ss.offer_sdp)

with b1:
    if st.button("Test-1", use_container_width=True):
        if _need_session():
            st.warning("Start a session first.")
        else:
            send_echo(ss.session_id, ss.session_token,
                      "Hello. Welcome to the test demonstration.")

# ----- Voice Start / Stop (Echo) with placeholder teardown -----

# where we mount the webrtc component
if "webrtc_slot" not in st.session_state:
    st.session_state.webrtc_slot = st.empty()
if "webrtc_ctx" not in st.session_state:
    st.session_state.webrtc_ctx = None

with b2:
    if st.button("Voice Start", use_container_width=True):
        ss["voice_run"] = True
        debug("[voice] start requested")

with b3:
    if st.button("Voice Stop", use_container_width=True):
        ss["voice_run"] = False
        # Remove the component from the page — this cleanly stops the stream
        try:
            st.session_state.webrtc_slot.empty()
        except Exception:
            pass
        st.session_state.webrtc_ctx = None
        debug("[voice] stop requested")

with b4:
    if st.button("Reset", use_container_width=True):
        for k in ("session_id","session_token","offer_sdp","rtc_config","voice_run"):
            ss[k] = None
        ss.debug_buf.clear()
        st.rerun()

with b5:
    st.write("")

# ----------------- Voice Echo -----------------
# Uses streamlit-webrtc to capture mic; simple pause-based chunking; Whisper (if key present).
if _HAS_WEBRTC and ss.get("voice_run") and ss.session_id and ss.session_token:
    import wave, tempfile
    from threading import Lock

    class EchoProcessor(AudioProcessorBase):
        """
        Accumulates PCM16; when ~800ms pause or 4s max buffer,
        flush -> transcribe -> speak back via streaming.task (repeat).
        """
        def __init__(self, speak_cb):
            self.sample_rate = 16000
            self.buf = bytearray()
            self.last_activity = time.time()
            self.lock = Lock()
            self.speak_cb = speak_cb

        def _flush_if_needed(self, force=False):
            now = time.time()
            inactive = (now - self.last_activity) > 0.8  # pause
            longbuf = len(self.buf) > (self.sample_rate * 2 * 4)  # 4s @16k, 16-bit
            if force or (inactive and len(self.buf) > self.sample_rate * 2 * 0.8) or longbuf:
                pcm = bytes(self.buf)
                self.buf.clear()
                text = transcribe_whisper(pcm, self.sample_rate)
                if text:
                    debug(f"[stt] {text}")
                    try:
                        self.speak_cb(text)
                    except Exception as e:
                        debug(f"[echo error] {e}")

        def recv_audio(self, frame):
            try:
                pcm16 = frame.to_ndarray(format="s16")
                # mono
                if pcm16.ndim == 2 and pcm16.shape[0] > 1:
                    pcm16 = pcm16[0:1, :]
                pcm16 = np.squeeze(pcm16).astype(np.int16)
                in_rate = frame.sample_rate

                # rudimentary VAD by RMS
                rms = float(np.sqrt(np.mean((pcm16.astype(np.float32))**2)))
                if rms > 200:  # voiced-ish
                    self.last_activity = time.time()

                # resample if needed
                if in_rate != self.sample_rate:
                    dur = pcm16.shape[0] / in_rate
                    new_len = int(dur * self.sample_rate)
                    # numpy.interp over sample index
                    pcm16 = np.interp(
                        np.linspace(0, pcm16.shape[0] - 1, new_len),
                        np.arange(pcm16.shape[0]),
                        pcm16.astype(np.float32),
                    ).astype(np.int16)

                with self.lock:
                    self.buf += pcm16.tobytes()
                    self._flush_if_needed(force=False)
            except Exception as e:
                debug(f"[audio] {e}")
            return frame

        def destroy(self):
            # final flush
            try:
                with self.lock:
                    self._flush_if_needed(force=True)
            except Exception:
                pass

    def transcribe_whisper(pcm_bytes: bytes, rate: int) -> Optional[str]:
        if not OPENAI_API_KEY or not pcm_bytes:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            # write temporary wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm_bytes)
            with open(path, "rb") as f:
                resp = client.audio.transcriptions.create(model="whisper-1", file=f)
            return (getattr(resp, "text", "") or "").strip() or None
        except Exception as e:
            debug(f"[whisper] {e}")
            return None

    def speak_back(text: str):
        # call HeyGen repeat task
        send_echo(ss.session_id, ss.session_token, text)

    # bind a processor instance (avoid referencing session_state inside worker thread)
    processor_instance = EchoProcessor(speak_back)

    # Mount inside the placeholder; store context if you want to inspect it
    with st.session_state.webrtc_slot.container():
        st.session_state.webrtc_ctx = webrtc_streamer(
            key="mic-echo",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=lambda: processor_instance,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=False,
        )
    st.caption("Voice echo is running…")

# -------------- Debug box (disabled) --------------
st.text_area("Debug", value="\n".join(ss.debug_buf), height=200, disabled=True)
