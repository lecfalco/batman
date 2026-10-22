"""Talking to Gemma 4. Three options, picked with the FL_BACKEND setting:
  ollama  - Gemma running on your own laptop (default). Needs `ollama pull gemma4:e4b`.
  gemini  - Gemma through Google's online API. Needs GEMINI_API_KEY.
  mock    - a simple rule-based stand-in so the whole project can be tested with no model.
            It is NOT Gemma, and every result made with it is labelled "offline stand-in".
We ask for answers as JSON (a strict, machine-readable format) and check them in code.

Extra Gemma features used here (all optional, all UNVERIFIED on your exact model until you run selftest.py):
  think=True      ask Ollama for "thinking mode" (step-by-step reasoning before the answer)
  chat_tools()    Ollama's native tool-calling API (instead of asking for tool calls as JSON text)
  temperature     the jury asks the same question several times with different random seeds"""
import base64, json, os, re, time, urllib.error, urllib.request

def _extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None

def _post(url, body, timeout=300):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def _env_flag(name):
    v = os.environ.get(name)
    return None if v is None else v.strip().lower() in ("1", "true", "yes", "on")

class Gemma:
    def __init__(self, backend=None, model=None, think=None):
        self.backend = backend or os.environ.get("FL_BACKEND", "ollama")
        if self.backend == "gemini":
            # Check the exact Gemma 4 model name in Google AI Studio and set FL_MODEL if this differs.
            self.model = model or os.environ.get("FL_MODEL", "gemma-4-31b-it")
        else:
            self.model = model or os.environ.get("FL_MODEL", "gemma4:e4b")
        self.think = think if think is not None else _env_flag("FL_THINK")
        self.think_ok = True                       # flips to False if the server rejects the think option
        base = f"{self.backend}:{self.model}"
        self.name = ("offline stand-in (not Gemma)" if self.backend == "mock"
                     else base + ("+think" if self.think else ""))
        self.calls = 0
        self.failures = 0

    def json(self, system, user, images=None, retries=3, temperature=0.0, seed=7):
        """Ask for a JSON answer. Retries if the model returns broken JSON. Returns dict or None."""
        for attempt in range(retries):
            self.calls += 1
            try:
                out = _extract_json(self._ask(system, user, images or [], temperature, seed + attempt))
            except Exception:
                out = None
            if isinstance(out, dict):
                return out
            self.failures += 1
            time.sleep(0.5)
        return None

    def _ask(self, system, user, images, temperature=0.0, seed=7):
        if self.backend == "ollama":
            msg = {"role": "user", "content": user}
            if images:
                msg["images"] = [base64.b64encode(open(p, "rb").read()).decode() for p in images]
            body = {"model": self.model, "stream": False, "format": "json",
                    "options": {"temperature": temperature, "seed": seed},
                    "messages": [{"role": "system", "content": system}, msg]}
            url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
            if self.think is not None and self.think_ok:
                try:
                    return _post(url, dict(body, think=bool(self.think)))["message"]["content"]
                except urllib.error.HTTPError:        # this model/server does not accept "think"
                    self.think_ok = False
            return _post(url, body)["message"]["content"]
        if self.backend == "gemini":
            parts = [{"text": system + "\n\n" + user + "\n\nReply with JSON only."}]
            for p in images:
                parts.append({"inline_data": {"mime_type": "image/png" if p.lower().endswith("png") else "image/jpeg",
                                              "data": base64.b64encode(open(p, "rb").read()).decode()}})
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
                   f"?key={os.environ['GEMINI_API_KEY']}")
            r = _post(url, {"contents": [{"role": "user", "parts": parts}],
                            "generationConfig": {"temperature": temperature}})
            return r["candidates"][0]["content"]["parts"][0]["text"]
        from faultlines.mock import mock_answer
        return json.dumps(mock_answer(system, user, images, {"temperature": temperature, "seed": seed}))

    def chat_tools(self, messages, tools):
        """Native tool calling (Ollama only). Returns the assistant message dict, or None if unavailable.
        If the message has "tool_calls", we run them; if not, the model thinks it has enough evidence."""
        if self.backend != "ollama":
            return None
        self.calls += 1
        try:
            r = _post(os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat",
                      {"model": self.model, "stream": False, "messages": messages, "tools": tools,
                       "options": {"temperature": 0, "seed": 7}})
            return r["message"]
        except Exception:
            self.failures += 1
            return None
