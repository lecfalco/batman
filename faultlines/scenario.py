"""Gemma reads a security advisory (typed text OR a screenshot) and we turn it into an epicenter."""
import re

SYSTEM = """TASK: read-advisory
You read a security advisory or bug report (text and/or a screenshot). Reply with JSON only:
{"package": "<name of the affected project or library>", "module_hint": "<a file or module name if one is mentioned, else empty>", "kind": "crash" | "data-leak" | "code-execution" | "other", "quote": "<short text copied from the advisory>"}"""

def read_advisory(model, world, text="", image=None):
    out = model.json(SYSTEM, text or "See the attached screenshot.", images=[image] if image else None)
    if not out:
        return {"ok": False, "reason": "model gave no usable answer"}
    seeds = []
    for key in (out.get("module_hint"), out.get("package")):
        if key:
            hint = re.sub(r"[^\w/\.]", "", str(key).replace(".", "/"))
            seeds = [m for m in world._resolve(hint)] or [m for m in world._resolve(str(key).lower())]
            if seeds:
                break
    return {"ok": bool(seeds), "seeds": seeds, "advisory": out, "model": model.name,
            "reason": "" if seeds else "the named package is not part of this map"}
