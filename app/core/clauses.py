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


class ClauseBuffer:
    def __init__(self, first_min: int = 15, soft_cap: int = 70, hard_cap: int | None = None):
        self.buf = ""
        self.first_done = False
        self.first_min = first_min
        self.soft_cap = soft_cap
        self.hard_cap = settings.clause_max_len if hard_cap is None else hard_cap

    def _term_cut(self) -> int | None:
        for match in re.finditer(r"[.!?]+(?=\s)", self.buf):
            i = match.end()
            toks = self.buf[:i].split()
            tail = re.sub(r"[.!?]+$", "", toks[-1]).rstrip(".").lower() if toks else ""
            if tail not in ABBREV:
                return i
        return None

    def _soft_cut(self, min_start: int) -> int | None:
        for match in re.finditer(r"[,;:](?=\s)", self.buf):
            if match.start() >= min_start:
                return match.end()
        return None

    def _find_cut(self) -> int | None:
        cut = self._term_cut()
        if not self.first_done:
            candidates = [i for i in (cut, self._soft_cut(self.first_min)) if i is not None]
            return min(candidates) if candidates else None
        if cut is not None:
            return cut
        if len(self.buf) >= self.soft_cap and (cut := self._soft_cut(0)) is not None:
            return cut
        if len(self.buf) >= self.hard_cap and (space := self.buf.rfind(" ", 0, self.hard_cap)) > 0:
            return space + 1
        return None

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            clause = self.buf[:cut].strip()
            self.buf = self.buf[cut:].lstrip()
            if clause:
                self.first_done = True
                out.append(clause)
        return out

    def flush(self) -> list[str]:
        clause = self.buf.strip()
        self.buf = ""
        if not clause:
            return []
        self.first_done = True
        return [clause]
