#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision (Вижен) — голосовой ИИ-ассистент для Mac в стиле Джарвиса.

Скажи «Вижен» (или «Vision») и команду: «Вижен, включи музыку», «Вижен, какая погода»,
«Вижен, открой Telegram», «Вижен, сделай громкость 30». После ответа он ещё несколько
секунд слушает без имени, так что можно просто продолжать разговор.

Запуск:  python3 vision.py          — всегда слушает имя
         python3 vision.py --enter  — говорить после нажатия Enter
         python3 vision.py --setup  — заново ввести ключи и настройки
"""
import io, os, re, sys, json, time, wave, queue, shutil, getpass, datetime, threading, subprocess, tempfile, urllib.parse
from collections import deque

try:
    import numpy as np
    import requests
    import sounddevice as sd
except ImportError:
    print("Не хватает библиотек. Запусти установку заново: pip install sounddevice numpy requests")
    sys.exit(1)

HOME = os.path.expanduser("~/Vision")
CFG_PATH = os.path.join(HOME, "config.json")
DEFAULT = {
    "name": "Вижен",
    "gemini_key": "",
    "anthropic_key": "",
    "openai_key": "",
    "brain": "gemini",
    "gemini_model": "gemini-2.5-flash",
    "gemini_endpoint": "ai",
    "stt": "local",
    "local_model": "mlx-community/whisper-large-v3-turbo",
    "tts": "edge",
    "model": "claude-haiku-5-5",
    "stt_model": "gpt-4o-mini-transcribe",
    "voice": "onyx",
    "city": "Алматы",
    "web_search": True,
    "follow_up_seconds": 6,
    "vocab": "",
}

# ---------------------------------------------------------------- terminal
C = {"dim": "\033[2m", "b": "\033[1m", "gold": "\033[33m", "blue": "\033[36m", "red": "\033[31m", "x": "\033[0m"}
def say_line(tag, text, color="dim"):
    print(f"{C[color]}{tag:>7}{C['x']}  {text}", flush=True)

# ---------------------------------------------------------------- config
def load_cfg():
    cfg = dict(DEFAULT)
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            raw = f.read()
        raw = raw.replace("\u201c", '"').replace("\u201d", '"')   # TextEdit "smart quotes"
        cfg.update(json.loads(raw))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"{C['red']}Файл настроек {CFG_PATH} повреждён: {e}. Запусти: ~/Vision/Vision.command --setup{C['x']}")
    for k in ("gemini_key", "anthropic_key", "openai_key"):
        cfg[k] = re.sub(r"\s+", "", str(cfg.get(k, "")))
    if cfg["openai_key"].startswith("sk-ant-") and not cfg["anthropic_key"].startswith("sk-ant-"):
        cfg["anthropic_key"], cfg["openai_key"] = cfg["openai_key"], cfg["anthropic_key"]   # keys swapped
    if cfg["openai_key"].startswith("sk-ant-"):
        print(f"{C['red']}В поле OpenAI стоит ключ Anthropic (sk-ant-…). Нужен ключ OpenAI с platform.openai.com/api-keys.{C['x']}")
        cfg["openai_key"] = ""
    return cfg

def save_cfg(cfg):
    os.makedirs(HOME, exist_ok=True)
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.chmod(CFG_PATH, 0o600)

GEMINI_HOSTS = {
    "ai": "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
    "vertex": "https://aiplatform.googleapis.com/v1/publishers/google/models/{m}:generateContent",
}

def check_gemini(key):
    """Tries the key on Google AI Studio and on Vertex AI (keys like AQ.…). Returns (ok, message, endpoint)."""
    order = ["vertex", "ai"] if key.startswith("AQ.") else ["ai", "vertex"]
    msg = ""
    for ep in order:
        for m in ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"):
            try:
                r = requests.post(GEMINI_HOSTS[ep].format(m=m), headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                  json={"contents": [{"role": "user", "parts": [{"text": "ok"}]}]}, timeout=30)
            except Exception as e:
                msg = str(e); continue
            if r.status_code == 200:
                return True, "", ep, m
            if r.status_code == 429:
                return True, "лимит на минуту, но ключ рабочий", ep, m
            try: msg = r.json().get("error", {}).get("message", r.text[:200])
            except Exception: msg = r.text[:200]
            if r.status_code != 404:
                break
    return False, msg, None, None

def setup(cfg, force=False):
    tty_in = sys.stdin
    if not sys.stdin.isatty():
        try: tty_in = open("/dev/tty")
        except Exception: pass
    def ask(prompt):
        print(prompt, end="", flush=True)
        return tty_in.readline().strip()
    if force or not cfg["gemini_key"]:
        print(f"\n{C['b']}Бесплатный ключ Google Gemini{C['x']} — мозг ассистента. Карта не нужна.")
        print("  1) Открой https://aistudio.google.com/apikey и войди через Google")
        print("  2) Нажми «Create API key» и скопируй ключ (начинается с AIza...)")
        while True:
            k = re.sub(r"\s+", "", ask("Вставь ключ Gemini и нажми Enter (Enter без ключа — пропустить): "))
            if not k: break
            ok, msg, ep, model = check_gemini(k)
            if ok:
                cfg["gemini_key"], cfg["brain"], cfg["gemini_endpoint"], cfg["gemini_model"] = k, "gemini", ep, model
                print(f"{C['gold']}  ✔ Ключ Gemini работает{C['x']}"); break
            print(f"{C['red']}  ✖ Google не принял ключ: {msg[:160]}{C['x']}")
    if force:
        c = ask(f"Город для погоды [{cfg['city']}]: ")
        if c: cfg["city"] = c
        v = ask(f"Слова, которые ты часто говоришь (имена, казахские слова), через запятую [{cfg['vocab']}]: ")
        if v: cfg["vocab"] = v
    if not cfg["gemini_key"] and cfg["anthropic_key"]:
        cfg["brain"] = "anthropic"
    save_cfg(cfg)
    if not cfg["gemini_key"] and not cfg["anthropic_key"]:
        print(f"{C['red']}Нужен ключ Gemini (бесплатно). Запусти: ~/Vision/Vision.command --setup{C['x']}")
        sys.exit(1)

CFG = load_cfg()

# ---------------------------------------------------------------- wake word by sound
CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщыэюяәғқңөұүһіьъ",
    ["a","b","v","g","d","e","e","zh","z","i","i","k","l","m","n","o","p","r","s","t","u","f","h","ts","ch","sh","sh","i","e","yu","ya","a","g","k","n","o","u","u","h","i","",""]))
WAKE_WORDS = {"вижен", "вижн", "визион", "vision", "visions", "visio", "fashion", "фэшн", "фешн", "вижин", "вижон", "виждн"}
WAKE_STOP = {"весна", "весну", "весне", "весной", "весны", "визе", "визу", "fusion", "вызов"}

def sound_key(w):
    w = "".join(CYR.get(ch, ch) for ch in w.lower())
    w = re.sub(r"[^a-z]", "", w).replace("ph", "f")
    w = re.sub(r"(zh|sh|ch|ss|s|z|j|dg|g(?=[ei]))", "z", w)
    w = re.sub(r"[fw]", "v", w)
    w = re.sub(r"[aeiouy]", "", w)
    return re.sub(r"(.)\1+", r"\1", w)

def wake_match(text, name):
    """Returns the words after the wake word, '' if only the name was said, None if not called."""
    tokens = text.strip().split()
    keys = {"vzn", sound_key(name)}
    for i, tok in enumerate(tokens[:3]):
        w = re.sub(r"[^\w-]", "", tok.lower())
        if not w or w in WAKE_STOP:
            continue
        if w in WAKE_WORDS or w == name.lower() or (len(w) <= 9 and sound_key(w) in keys):
            return " ".join(tokens[i + 1:]).strip(" ,.!?—-")
    return None

# ---------------------------------------------------------------- microphone
RATE, FRAME = 16000, 480            # 30 ms frames
FRAME_SEC = FRAME / RATE

class Mic:
    def __init__(self):
        self.q = queue.Queue()
        self.muted = False
        self.floor = 300.0
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME, callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        if not self.muted:
            self.q.put(indata[:, 0].copy())

    def flush(self):
        try:
            while True: self.q.get_nowait()
        except queue.Empty:
            pass

    @property
    def threshold(self):
        return max(500.0, self.floor * 3.2)

    def calibrate(self, seconds=1.0):
        vals = []
        end = time.time() + seconds
        while time.time() < end:
            try: vals.append(rms(self.q.get(timeout=1)))
            except queue.Empty: break
        if vals:
            self.floor = float(np.median(vals))

    def record(self, start_timeout=None, end_silence=0.9, max_len=15.0):
        """Waits for speech, returns the utterance as int16 array, or None on timeout."""
        pre = deque(maxlen=12)
        frames, voiced, silent, started, loud_n = [], 0, 0, False, 0
        t0 = time.time()
        while True:
            try:
                f = self.q.get(timeout=0.5)
            except queue.Empty:
                if start_timeout and not started and time.time() - t0 > start_timeout:
                    return None
                continue
            r = rms(f)
            loud = r > self.threshold
            if not started:
                pre.append(f)
                if not loud:
                    self.floor = 0.97 * self.floor + 0.03 * r
                voiced = voiced + 1 if loud else 0
                if voiced >= 3:
                    started, frames, silent, loud_n = True, list(pre), 0, voiced
                elif start_timeout and time.time() - t0 > start_timeout:
                    return None
            else:
                frames.append(f)
                silent = 0 if loud else silent + 1
                loud_n += 1 if loud else 0
                if silent * FRAME_SEC >= end_silence or len(frames) * FRAME_SEC >= max_len:
                    audio = np.concatenate(frames)
                    if loud_n * FRAME_SEC < 0.3:
                        frames, voiced, started = [], 0, False
                        pre.clear()
                        continue
                    return audio

def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

def to_wav(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    return buf.getvalue()

# ---------------------------------------------------------------- hearing (OpenAI)
HALLUCINATIONS = re.compile(r"(продолжение следует|субтитр|спасибо за просмотр|редактор|подписывайтесь|thank(s| you) for watching|amara\.org|dimatorzok)", re.I)
_LOCAL_OK = None

def _stt_local(audio, prompt):
    import mlx_whisper
    out = mlx_whisper.transcribe(audio.astype(np.float32) / 32768.0, path_or_hf_repo=CFG["local_model"],
                                 initial_prompt=prompt, condition_on_previous_text=False, verbose=None)
    return (out.get("text") or "").strip()

def transcribe(audio):
    global _LOCAL_OK
    prompt = (f"Говорящий из Казахстана, говорит по-русски, по-казахски или по-английски. Ассистента зовут {CFG['name']}."
              + (" Слова: " + CFG["vocab"].strip() + "." if CFG["vocab"].strip() else ""))
    if CFG["stt"] == "local" and _LOCAL_OK is not False:
        try:
            text = _stt_local(audio, prompt); _LOCAL_OK = True
            return "" if HALLUCINATIONS.search(text) else text
        except ImportError:
            _LOCAL_OK = False
            say_line("!", "локальное распознавание недоступно на этом Mac — пробую другой способ", "red")
    if CFG["openai_key"]:
        return _stt_openai(audio)
    if CFG["gemini_key"]:
        return _stt_gemini(audio, prompt)
    raise requests.RequestException("нет способа распознать речь")

def _stt_gemini(audio, prompt):
    import base64
    body = {"contents": [{"role": "user", "parts": [
        {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(to_wav(audio)).decode()}},
        {"text": prompt + " Запиши дословно, что сказано в аудио, без комментариев. Если речи нет — верни пустую строку."}]}]}
    data = gemini_call(body)
    return "".join(p.get("text", "") for p in data["candidates"][0]["content"].get("parts", [])).strip()

def _stt_openai(audio):
    prompt = (f"Говорящий из Казахстана, говорит по-русски, по-казахски (қазақша) или по-английски, иногда смешивая. "
              f"Казахские слова пиши казахской кириллицей. Ассистента зовут {CFG['name']} (Vision).")
    if CFG["vocab"].strip():
        prompt += " Частые слова: " + CFG["vocab"].strip() + "."
    r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                      headers={"Authorization": "Bearer " + CFG["openai_key"]},
                      files={"file": ("speech.wav", to_wav(audio), "audio/wav")},
                      data={"model": CFG["stt_model"], "prompt": prompt}, timeout=40)
    if r.status_code == 401:
        try: msg = r.json()["error"]["message"]
        except Exception: msg = r.text[:200]
        print(f"{C['red']}OpenAI не принял ключ: {msg}\nИсправь ключ: ~/Vision/Vision.command --setup{C['x']}")
        sys.exit(1)
    if r.status_code == 429:
        raise requests.RequestException("OpenAI: закончились деньги на балансе или слишком много запросов (429). Пополни баланс на platform.openai.com → Billing.")
    r.raise_for_status()
    return (r.json().get("text") or "").strip()

# ---------------------------------------------------------------- voice
speak_lock = threading.Lock()
MIC = None

def detect_lang(s):
    if re.search(r"[әғқңөұүһі]", s, re.I): return "kk"
    if re.search(r"[а-яё]", s, re.I): return "ru"
    return "en"

def mac_voice(lang):
    try:
        voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    except Exception:
        return None
    prefs = {"ru": ["Yuri", "Milena"], "kk": ["Yuri", "Milena"], "en": ["Daniel", "Alex"]}[lang]
    for p in prefs:
        if re.search(rf"^{p}\b", voices, re.M):
            return p
    return None

def speak(text):
    text = clean(text)
    if not text:
        return
    say_line(CFG["name"], text, "blue")
    with speak_lock:
        if MIC: MIC.muted = True
        try:
            ok = False
            if CFG["tts"] == "edge": ok = _speak_edge(text)
            elif CFG["tts"] == "openai" and CFG["openai_key"]: ok = _speak_openai(text)
            if not ok:
                v = mac_voice(detect_lang(text))
                subprocess.run(["say"] + (["-v", v] if v else []) + [text])
        finally:
            time.sleep(0.25)
            if MIC:
                MIC.flush(); MIC.muted = False

EDGE_VOICES = {"ru": "ru-RU-DmitryNeural", "kk": "kk-KZ-DauletNeural", "en": "en-GB-RyanNeural"}
def _speak_edge(text):
    try:
        import asyncio, edge_tts
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f: path = f.name
        asyncio.run(edge_tts.Communicate(text, EDGE_VOICES[detect_lang(text)], rate="-5%", pitch="-4Hz").save(path))
        subprocess.run(["afplay", path]); os.unlink(path)
        return True
    except Exception:
        return False

def _speak_openai(text):
    lang = {"ru": "Russian", "kk": "Kazakh, with natural native Kazakh pronunciation", "en": "English"}[detect_lang(text)]
    try:
        r = requests.post("https://api.openai.com/v1/audio/speech",
            headers={"Authorization": "Bearer " + CFG["openai_key"], "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini-tts", "voice": CFG["voice"], "input": text, "response_format": "mp3",
                  "instructions": f"Speak {lang}. You are a refined AI butler: calm, measured, confident and warm, "
                                  "with a light, dry sense of humour. Clear diction, unhurried pace, never robotic."},
            timeout=40)
        if r.status_code != 200:
            return False
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(r.content); path = f.name
        subprocess.run(["afplay", path]); os.unlink(path)
        return True
    except Exception:
        return False

def chime():
    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

def clean(s):
    s = re.sub(r"[*_#`>]+", "", str(s))
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------- Mac actions
def osa(script):
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "AppleScript error")
    return p.stdout.strip()

def q(s):  # AppleScript string literal
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'

def app_running(name):
    try: return osa(f'application {q(name)} is running') == "true"
    except Exception: return False

def t_open_app(i):
    name = str(i.get("name", "")).strip()
    p = subprocess.run(["open", "-a", name], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"app not found: {name}")
    return f"opened {name}"

def t_quit_app(i):
    osa(f'quit app {q(i.get("name", ""))}')
    return "quit"

def t_open_website(i):
    url = str(i.get("url", "")).strip()
    if not re.match(r"^https?://", url): url = "https://" + url
    subprocess.run(["open", url]); return "opened"

def t_google(i):
    subprocess.run(["open", "https://www.google.com/search?q=" + urllib.parse.quote(str(i.get("query", ""))) ]); return "opened"

def youtube_first(query):
    r = requests.get("https://www.youtube.com/results", params={"search_query": query},
                     headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "ru,en"}, timeout=15)
    m = re.search(r'"videoId":"([\w-]{11})"', r.text)
    return m.group(1) if m else None

def t_play_music(i):
    query = str(i.get("query", "")).strip()
    service = str(i.get("service", "youtube")).lower()
    if service == "apple_music":
        if query:
            try:
                osa(f'tell application "Music" to play (first track of library playlist 1 whose name contains {q(query)} or artist contains {q(query)})')
                return f"playing '{query}' from the Music library"
            except Exception:
                pass
        else:
            osa('tell application "Music" to play'); return "Music playing"
    if service == "spotify":
        subprocess.run(["open", "spotify:search:" + urllib.parse.quote(query)]); return "opened Spotify search"
    vid = youtube_first(query or "музыка")
    if vid:
        subprocess.run(["open", f"https://www.youtube.com/watch?v={vid}"]); return f"playing on YouTube: {query}"
    subprocess.run(["open", "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)]); return "opened YouTube search"

def t_media(i):
    action = str(i.get("action", "playpause"))
    cmd = {"play": "play", "pause": "pause", "playpause": "playpause", "next": "next track", "previous": "previous track"}.get(action, "playpause")
    done = []
    for app in ("Music", "Spotify"):
        if app_running(app):
            osa(f'tell application "{app}" to {cmd}'); done.append(app)
    if not done:
        raise RuntimeError("Music and Spotify are not running; for YouTube in the browser ask the user to use the page")
    return f"{action} in {', '.join(done)}"

def t_volume(i):
    if i.get("mute") is True:
        osa("set volume with output muted"); return "muted"
    cur = int(osa("output volume of (get volume settings)") or 50)
    if "level" in i and i["level"] is not None:
        new = int(i["level"])
    else:
        new = cur + int(i.get("change", 10))
    new = max(0, min(100, new))
    osa(f"set volume output volume {new} without output muted")
    return f"volume {new}"

def t_time(i):
    now = datetime.datetime.now()
    out = {"local": now.strftime("%Y-%m-%d %H:%M"), "weekday": now.strftime("%A")}
    tz = i.get("timezone")
    if tz:
        try:
            from zoneinfo import ZoneInfo
            out["other"] = datetime.datetime.now(ZoneInfo(str(tz))).strftime("%Y-%m-%d %H:%M"); out["timezone"] = tz
        except Exception:
            raise RuntimeError("unknown timezone")
    return out

WMO = [(0, "ясно"), (2, "переменная облачность"), (3, "пасмурно"), (48, "туман"), (57, "морось"), (67, "дождь"),
       (77, "снег"), (82, "ливень"), (86, "снегопад"), (99, "гроза")]
def wmo(c):
    for limit, name in WMO:
        if c <= limit: return name
    return "?"

def t_weather(i):
    city = str(i.get("city") or CFG["city"])
    days = max(1, min(7, int(i.get("days") or 1)))
    g = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "ru"}, timeout=15).json()
    if not g.get("results"): raise RuntimeError("city not found: " + city)
    p = g["results"][0]
    w = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto", "forecast_days": days + 1,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"}, timeout=15).json()
    c, d = w["current"], w["daily"]
    return {"place": p["name"], "now": {"temp_c": c["temperature_2m"], "feels_c": c["apparent_temperature"], "wind_kmh": c["wind_speed_10m"], "sky": wmo(c["weather_code"])},
            "days": [{"date": d["time"][k], "min_c": d["temperature_2m_min"][k], "max_c": d["temperature_2m_max"][k],
                      "rain_chance": d["precipitation_probability_max"][k], "sky": wmo(d["weather_code"][k])} for k in range(len(d["time"]))]}

TIMERS = {}
def t_timer(i):
    secs = int(i.get("seconds", 0)); label = str(i.get("label", "")).strip()
    if not 0 < secs <= 86400: raise RuntimeError("seconds must be 1..86400")
    tid = str(int(time.time() * 1000))[-6:]
    def ring():
        TIMERS.pop(tid, None)
        for _ in range(3):
            subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])
        try: osa(f'display notification {q(label or "Время вышло")} with title {q(CFG["name"])} sound name "Glass"')
        except Exception: pass
        speak(f"Сэр, таймер{(' «' + label + '»') if label else ''} закончился.")
    t = threading.Timer(secs, ring); t.daemon = True; t.start()
    TIMERS[tid] = (t, label, time.time() + secs)
    return {"id": tid, "ends_at": datetime.datetime.fromtimestamp(time.time() + secs).strftime("%H:%M:%S")}

def t_list_timers(i):
    return [{"id": k, "label": v[1], "seconds_left": int(v[2] - time.time())} for k, v in TIMERS.items()]

def t_cancel_timer(i):
    t = TIMERS.pop(str(i.get("id")), None)
    if not t: raise RuntimeError("not found")
    t[0].cancel(); return "cancelled"

def t_reminder(i):
    text = str(i.get("text", "")).strip()
    when = None
    if i.get("in_minutes") is not None:
        when = datetime.datetime.now() + datetime.timedelta(minutes=float(i["in_minutes"]))
    elif i.get("at_local"):
        when = datetime.datetime.strptime(str(i["at_local"])[:16], "%Y-%m-%dT%H:%M")
    props = f"{{name:{q(text)}}}"
    if when:
        script = (f'set d to current date\nset year of d to {when.year}\nset month of d to {when.month}\nset day of d to {when.day}\n'
                  f'set hours of d to {when.hour}\nset minutes of d to {when.minute}\nset seconds of d to 0\n'
                  f'tell application "Reminders" to make new reminder with properties {{name:{q(text)}, remind me date:d}}')
    else:
        script = f'tell application "Reminders" to make new reminder with properties {props}'
    osa(script)
    return {"saved_to": "Apple Reminders (syncs to iPhone)", "at": when.strftime("%Y-%m-%d %H:%M") if when else None}

def t_note(i):
    text = str(i.get("text", "")).strip()
    title = text.split("\n")[0][:60]
    osa(f'tell application "Notes" to make new note with properties {{name:{q(title)}, body:{q(text)}}}')
    return "saved to Apple Notes (syncs to iPhone)"

def t_system(i):
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)%;\s*([^;]+)", batt)
    du = shutil.disk_usage("/")
    return {"battery_percent": int(m.group(1)) if m else None, "battery_state": m.group(2).strip() if m else None,
            "disk_free_gb": round(du.free / 1e9, 1)}

def t_screenshot(i):
    path = os.path.expanduser(f"~/Desktop/Vision {datetime.datetime.now():%Y-%m-%d %H.%M.%S}.png")
    subprocess.run(["screencapture", "-x", path]); return {"saved": path}

def t_sleep_display(i):
    subprocess.run(["pmset", "displaysleepnow"]); return "display off"

def t_shutdown_assistant(i):
    threading.Timer(4, lambda: os._exit(0)).start(); return "assistant will exit after this reply"

TOOLS = [
    ("open_app", "Open a Mac application by its English app name (Safari, Telegram, WhatsApp, Music, Spotify, Calculator, Finder, System Settings, Google Chrome, Visual Studio Code, Notes...).", t_open_app,
        {"name": {"type": "string"}}, ["name"]),
    ("quit_app", "Quit a running Mac application by name.", t_quit_app, {"name": {"type": "string"}}, ["name"]),
    ("open_website", "Open a website in the default browser.", t_open_website, {"url": {"type": "string"}}, ["url"]),
    ("google_search", "Open a Google search page in the browser for the user to look at.", t_google, {"query": {"type": "string"}}, ["query"]),
    ("play_music", "Play music or a video. service: youtube (default, plays the first match in the browser), apple_music (user's Music library), spotify (opens search). query = song/artist/genre/mood.",
        t_play_music, {"query": {"type": "string"}, "service": {"type": "string", "enum": ["youtube", "apple_music", "spotify"]}}, ["query"]),
    ("media_control", "Play, pause, next or previous track in the Music or Spotify app.", t_media,
        {"action": {"type": "string", "enum": ["play", "pause", "playpause", "next", "previous"]}}, ["action"]),
    ("set_volume", "Change system volume: level 0-100, or change (+/- steps), or mute true.", t_volume,
        {"level": {"type": "integer"}, "change": {"type": "integer"}, "mute": {"type": "boolean"}}, []),
    ("get_time", "Current local date/time, optionally time in an IANA timezone.", t_time, {"timezone": {"type": "string"}}, []),
    ("get_weather", "Weather now and forecast in Celsius. Omit city for the user's home city.", t_weather,
        {"city": {"type": "string"}, "days": {"type": "integer"}}, []),
    ("set_timer", "Start a countdown timer that rings and speaks when done.", t_timer,
        {"seconds": {"type": "integer"}, "label": {"type": "string"}}, ["seconds"]),
    ("list_timers", "List running timers.", t_list_timers, {}, []),
    ("cancel_timer", "Cancel a timer by id.", t_cancel_timer, {"id": {"type": "string"}}, ["id"]),
    ("create_reminder", "Create a reminder in Apple Reminders (it also appears on the user's iPhone). Give at_local YYYY-MM-DDTHH:MM or in_minutes, or neither for an undated one.",
        t_reminder, {"text": {"type": "string"}, "at_local": {"type": "string"}, "in_minutes": {"type": "number"}}, ["text"]),
    ("add_note", "Save a note to Apple Notes (also on the user's iPhone).", t_note, {"text": {"type": "string"}}, ["text"]),
    ("system_status", "Battery level and free disk space of this Mac.", t_system, {}, []),
    ("take_screenshot", "Take a screenshot and save it to the Desktop.", t_screenshot, {}, []),
    ("sleep_display", "Turn the display off.", t_sleep_display, {}, []),
    ("shutdown_assistant", "Exit the assistant program when the user says goodbye for good / tells you to turn off.", t_shutdown_assistant, {}, []),
]
TOOL_FN = {n: f for n, _, f, _, _ in TOOLS}
TOOL_SPEC = [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p, "required": r}} for n, d, _, p, r in TOOLS]

def run_tool(name, inp):
    say_line("⚙", f"{name} {json.dumps(inp, ensure_ascii=False)}", "gold")
    try:
        out = TOOL_FN[name](inp or {})
        return (out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)), False
    except Exception as e:
        return f"Error: {e}", True

# ---------------------------------------------------------------- brain (Claude)
HISTORY = []
WEB_OK = True

def system_prompt():
    now = datetime.datetime.now()
    hello = "доброе утро" if now.hour < 12 else "добрый день" if now.hour < 18 else "добрый вечер"
    vocab = f"\n- Слова, которые пользователь часто говорит: {CFG['vocab']}." if CFG["vocab"].strip() else ""
    return f"""Ты — {CFG['name']} (Vision), личный ИИ-ассистент пользователя, работающий на его Mac. Стиль — как у ИИ-дворецкого из фильмов про супергероев: спокойный, безупречно вежливый, уверенный, с лёгким сухим юмором. Обращаешься к пользователю «сэр» (по-казахски «мырза», по-английски "sir").
Сейчас {now:%Y-%m-%d %H:%M}, {now:%A}. Уместное приветствие сейчас: «{hello}». Город пользователя: {CFG['city']}.
Правила:
- Отвечай на языке пользователя: русский, казахский (казахской кириллицей) или английский.
- Ответ звучит вслух: 1–3 коротких фразы, без markdown, списков, эмодзи и ссылок. Числа — так, как их удобно произносить.
- Команды выполняй через инструменты сразу, не переспрашивая, если смысл понятен. После действия подтверждай коротко, в своём стиле: «Включаю, сэр.», «Готово, сэр. Громкость тридцать процентов.»
- Название приложения переводи в английское имя для macOS (Телеграм → Telegram, Сафари → Safari, Музыка → Music, Настройки → System Settings, Хром → Google Chrome).
- «Включи музыку» без уточнений — включи на YouTube что-нибудь подходящее по времени суток и настроению.
- Для погоды, времени в других городах, свежих новостей, курсов и спорта используй инструменты и поиск (search_internet); ничего не выдумывай.
- Текст пришёл из распознавания речи и может содержать ошибки: казахские слова русскими буквами, своё имя как «fashion» или «вижн». Догадывайся по звучанию и смыслу; переспрашивай одним коротким вопросом, только если совсем непонятно.
- Если сказали «спасибо, всё» или «отдыхай» — коротко попрощайся (программа продолжит ждать твоего имени). Если просят выключиться совсем — вызови shutdown_assistant.
- Отправлять сообщения, звонить, покупать и удалять файлы ты не умеешь: так и скажи.{vocab}"""

def ask_claude(text):
    global WEB_OK
    msgs = [dict(m) for m in HISTORY[-12:]]
    while msgs and msgs[0]["role"] != "user": msgs.pop(0)
    msgs.append({"role": "user", "content": text})
    reply = ""
    for _ in range(8):
        tools = TOOL_SPEC + ([{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}] if CFG["web_search"] and WEB_OK else [])
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": CFG["anthropic_key"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": CFG["model"], "max_tokens": 900, "system": system_prompt(), "tools": tools, "messages": msgs}, timeout=90)
        if r.status_code == 400 and WEB_OK and "web_search" in r.text:
            WEB_OK = False; continue
        if r.status_code == 401:
            return "Сэр, ключ Anthropic не подошёл. Запустите меня с параметром setup и введите ключ заново."
        r.raise_for_status()
        data = r.json()
        msgs.append({"role": "assistant", "content": data["content"]})
        if data.get("stop_reason") == "pause_turn":
            continue
        if any(b.get("type") == "server_tool_use" for b in data["content"]):
            say_line("🌐", "поиск в интернете", "gold")
        uses = [b for b in data["content"] if b.get("type") == "tool_use"]
        if data.get("stop_reason") != "tool_use" or not uses:
            reply = " ".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()
            break
        results = []
        for u in uses:
            out, err = run_tool(u["name"], u.get("input"))
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": out, "is_error": err})
        msgs.append({"role": "user", "content": results})
    reply = clean(reply) or "Готово, сэр."
    HISTORY.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
    del HISTORY[:-16]
    return reply

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]
def gemini_call(body):
    tried = [CFG["gemini_model"]] + [m for m in GEMINI_MODELS if m != CFG["gemini_model"]]
    last = ""
    for m in tried:
        r = requests.post(GEMINI_HOSTS.get(CFG.get("gemini_endpoint", "ai"), GEMINI_HOSTS["ai"]).format(m=m),
                          headers={"x-goog-api-key": CFG["gemini_key"], "Content-Type": "application/json"}, json=body, timeout=90)
        if r.status_code == 404:
            last = r.text[:200]; continue
        if r.status_code == 429:
            raise requests.RequestException("Gemini: исчерпан бесплатный лимит запросов, подожди минуту")
        if r.status_code in (400, 403) and "API key" in r.text:
            raise requests.RequestException("Gemini не принял ключ. Запусти ~/Vision/Vision.command --setup")
        r.raise_for_status()
        CFG["gemini_model"] = m
        return r.json()
    raise requests.RequestException("Gemini: модель не найдена " + last)

def _gschema(x):
    if isinstance(x, dict):
        return {k: (v.upper() if k == "type" and isinstance(v, str) else _gschema(v)) for k, v in x.items()}
    return [_gschema(i) for i in x] if isinstance(x, list) else x

def gemini_tools():
    decls = []
    for t in TOOL_SPEC + [SEARCH_SPEC]:
        d = {"name": t["name"], "description": t["description"]}
        if t["input_schema"]["properties"]:
            d["parameters"] = _gschema(t["input_schema"])
        decls.append(d)
    return [{"functionDeclarations": decls}]

def t_search(i):
    q = str(i.get("query", ""))
    data = gemini_call({"contents": [{"role": "user", "parts": [{"text": f"Найди в интернете актуальную информацию и ответь кратко (3-5 фактов, с цифрами и датами): {q}"}]}],
                        "tools": [{"googleSearch": {}}]})
    return "".join(p.get("text", "") for p in data["candidates"][0]["content"].get("parts", []))[:3000]

SEARCH_SPEC = {"name": "search_internet", "description": "Search the internet for current information: news, prices, exchange rates, sports results, schedules, facts about people and places. Returns a short summary.",
               "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
TOOL_FN["search_internet"] = t_search

def ask_gemini(text):
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in HISTORY[-12:]]
    while contents and contents[0]["role"] != "user": contents.pop(0)
    contents.append({"role": "user", "parts": [{"text": text}]})
    reply = ""
    for _ in range(6):
        data = gemini_call({"systemInstruction": {"parts": [{"text": system_prompt()}]}, "contents": contents,
                            "tools": gemini_tools(), "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.7}})
        cands = data.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts", [])
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            reply = " ".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought")).strip()
            break
        contents.append({"role": "model", "parts": parts})
        resp = []
        for c in calls:
            out, err = run_tool(c["name"], c.get("args") or {})
            resp.append({"functionResponse": {"name": c["name"], "response": {"error" if err else "result": out}}})
        contents.append({"role": "user", "parts": resp})
    reply = clean(reply) or "Готово, сэр."
    HISTORY.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
    del HISTORY[:-16]
    return reply

def ask_brain(text):
    if CFG["brain"] == "gemini" and CFG["gemini_key"]:
        return ask_gemini(text)
    return ask_claude(text)

def handle(text):
    say_line("Вы", text, "b")
    try:
        speak(ask_brain(text))
    except requests.RequestException as e:
        say_line("ошибка", str(e)[:200], "red")
        speak("Сэр, бесплатный лимит на минуту исчерпан. Дайте мне немного передохнуть." if "лимит" in str(e) else "Сэр, связь с сервером пропала. Попробуйте ещё раз.")

# ---------------------------------------------------------------- main loops
def conversation(mic, first_text):
    """One request, then keep listening a few seconds for follow-ups without the name."""
    text = first_text
    while text:
        handle(text)
        audio = mic.record(start_timeout=CFG["follow_up_seconds"], end_silence=1.1, max_len=30)
        if audio is None:
            break
        try:
            text = transcribe(audio)
        except requests.RequestException:
            break
        wm = wake_match(text, CFG["name"]) if text else None
        if wm == "":
            chime()
            audio = mic.record(start_timeout=8, end_silence=1.2, max_len=30)
            text = transcribe(audio) if audio is not None else None
        elif wm:
            text = wm
    say_line("", f"жду «{CFG['name']}»…")

def run_wake(mic):
    say_line("", f"жду «{CFG['name']}»… (Ctrl+C — выход)")
    while True:
        audio = mic.record(end_silence=0.8, max_len=8)
        if audio is None: continue
        try:
            heard = transcribe(audio)
        except requests.RequestException as e:
            say_line("ошибка", str(e)[:200], "red"); time.sleep(2); continue
        if not heard: continue
        rest = wake_match(heard, CFG["name"])
        if rest is None:
            say_line("·", heard); continue
        if rest:
            conversation(mic, rest)
        else:
            chime()
            audio = mic.record(start_timeout=8, end_silence=1.2, max_len=30)
            if audio is None:
                say_line("", "не расслышал команду"); continue
            text = transcribe(audio)
            if text: conversation(mic, text)

def run_enter(mic):
    tty = open("/dev/tty") if not sys.stdin.isatty() else sys.stdin
    while True:
        print(f"{C['dim']}Enter — говорить, Ctrl+C — выход{C['x']}", flush=True)
        tty.readline(); mic.flush(); chime()
        audio = mic.record(start_timeout=8, end_silence=1.2, max_len=30)
        if audio is None: continue
        text = transcribe(audio)
        if text: conversation(mic, text)

def main():
    global MIC
    setup(CFG, force="--setup" in sys.argv)
    print(f"\n{C['b']}{CFG['name']} (Vision){C['x']} — голосовой ассистент. Микрофон: {sd.query_devices(kind='input')['name']}")
    try:
        MIC = Mic()
    except Exception as e:
        print(f"{C['red']}Нет доступа к микрофону: {e}\nОткрой Системные настройки → Конфиденциальность и безопасность → Микрофон и разреши Терминалу.{C['x']}")
        sys.exit(1)
    MIC.calibrate()
    if CFG["stt"] == "local":
        say_line("", "загружаю распознавание речи (в первый раз скачается ~1.6 ГБ, подожди)…")
        try:
            _stt_local(np.zeros(RATE // 2, dtype=np.int16), "")
        except ImportError:
            pass
        except Exception as e:
            say_line("!", f"локальное распознавание: {e}", "red")
    h = datetime.datetime.now().hour
    speak(("Доброе утро" if 5 <= h < 12 else "Добрый день" if 12 <= h < 18 else "Добрый вечер") + f", сэр. {CFG['name']} на связи.")
    try:
        (run_enter if "--enter" in sys.argv else run_wake)(MIC)
    except KeyboardInterrupt:
        print("\nДо встречи, сэр.")

if __name__ == "__main__":
    main()
