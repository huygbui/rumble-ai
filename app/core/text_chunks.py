import re

from app.core.config import settings

ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "mt",
    "vs",
    "jr",
    "sr",
    "prof",
    "sgt",
    "gen",
    "rev",
    "hon",
    "co",
    "inc",
    "ltd",
    "etc",
    "e.g",
    "i.e",
    "a.m",
    "p.m",
}


class StreamingClauseSplitter:
    def __init__(self, first_min=15, soft_cap=70, hard_cap=140):
        self.buf = ""
        self.first_done = False
        self.first_min = first_min
        self.soft_cap = soft_cap
        self.hard_cap = hard_cap

    def _term_cut(self):
        for m in re.finditer(r"[.!?]+(?=\s)", self.buf):
            i = m.end()
            toks = self.buf[:i].split()
            tail = re.sub(r"[.!?]+$", "", toks[-1]).rstrip(".").lower() if toks else ""
            if tail in ABBREV:
                continue
            return i
        return None

    def _soft_cut(self, min_start):
        for m in re.finditer(r"[,;:](?=\s)", self.buf):
            if m.start() >= min_start:
                return m.end()
        return None

    def _find_cut(self):
        t = self._term_cut()
        if not self.first_done:
            c = self._soft_cut(self.first_min)
            cands = [x for x in (t, c) if x is not None]
            if cands:
                return min(cands)
        else:
            if t is not None:
                return t
            if len(self.buf) >= self.soft_cap:
                c = self._soft_cut(0)
                if c is not None:
                    return c
        if len(self.buf) >= self.hard_cap:
            sp = self.buf.rfind(" ", 0, self.hard_cap)
            if sp > 0:
                return sp + 1
        return None

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            clause = self.buf[:cut].strip()
            self.buf = self.buf[cut:].lstrip()
            if clause:
                out.append(clause)
                self.first_done = True
        return out

    def flush(self) -> list[str]:
        c = self.buf.strip()
        self.buf = ""
        if c:
            self.first_done = True
            return [c]
        return []


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for p in re.split(r"(?<=[.!?])\s+", text.strip()):
        p = p.strip()
        if not p:
            continue
        if out:
            toks = out[-1].split()
            tail = toks[-1].lower().rstrip(".") if toks else ""
            if tail in ABBREV:
                out[-1] = out[-1] + " " + p
                continue
        out.append(p)
    return out


def _merge_tiny(chunks: list[str], min_len: int | None = None) -> list[str]:
    min_len = settings.say_min_len if min_len is None else min_len
    out: list[str] = []
    for c in chunks:
        if out and len(out[-1]) < min_len:
            out[-1] = (out[-1] + " " + c).strip()
        else:
            out.append(c)
    if len(out) >= 2 and len(out[-1]) < min_len:
        out[-2] = (out[-2] + " " + out.pop()).strip()
    return out


def _shorten_first(chunks: list[str], first_max: int | None = None) -> list[str]:
    first_max = settings.say_first_max if first_max is None else first_max
    if not chunks or len(chunks[0]) <= first_max:
        return chunks
    head = chunks[0]
    m = next((mm for mm in re.finditer(r"[,;:]\s+", head) if mm.start() >= 20), None)
    if not m:
        return chunks
    first, rest = head[: m.start() + 1].strip(), head[m.end():].strip()
    return [first, rest] + chunks[1:]


def split_clauses(text: str, max_len: int | None = None) -> list[str]:
    max_len = settings.say_max_len if max_len is None else max_len
    chunks: list[str] = []
    for sent in _sentences(text):
        if len(sent) <= max_len:
            chunks.append(sent)
            continue
        buf = ""
        for part in re.split(r"(?<=[,;:])\s+", sent):
            if len(buf) + len(part) + 1 <= max_len:
                buf = (buf + " " + part).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = part
        if buf:
            chunks.append(buf)
    chunks = _merge_tiny(chunks)
    chunks = _shorten_first(chunks)
    return chunks
