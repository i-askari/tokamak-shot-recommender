"""Operator logbook parsing and keyword sentiment, attributed per shot.

Ported from notebooks/04_logbook_sentiment_analysis.ipynb.
"""

import re

SHOT_ENTRY_PATTERN = re.compile(r"Shot (\d+) ([A-Z_]+):")
RELEVANT_ROLES = ["SESSION_LEADER", "PHYSICS_OPERATOR", "CHIEF_OPERATOR", "DIAGNOSTICS", "BEAMS"]

# 'mode' and 'clean' are deliberately excluded: they false-match plasma
# physics vocabulary (O-mode, H-mode, wall cleaning).
POSITIVE_KEYWORDS = ["good", "ok", "okay", "runs", "better", "nice", "well",
                     "matched", "repeat", "success", "worked", "stable"]
NEGATIVE_KEYWORDS = ["trip", "abort", "missed", "issue", "problem", "too far",
                     "culprit", "bad", "failed", "disruption", "unfortunately", ":(", "worse"]


def clean_html(html_text):
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"&nbsp;", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def parse_session_html_by_shot(clean_text, relevant_roles=RELEVANT_ROLES):
    """Splits the electronic logbook into notes keyed by each entry's own shot number."""
    log_start = clean_text.find("ELECTRONIC LOGBOOK ENTRIES")
    if log_start == -1:
        return {}
    log_text = clean_text[log_start:]
    matches = list(SHOT_ENTRY_PATTERN.finditer(log_text))
    shot_notes = {}
    for i, m in enumerate(matches):
        shot_number, role = int(m.group(1)), m.group(2)
        if role not in relevant_roles:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(log_text)
        note_text = log_text[m.end():end].strip()
        if note_text:
            shot_notes.setdefault(shot_number, []).append(f"[{role}] {note_text}")
    return {shot: "\n\n".join(notes) for shot, notes in shot_notes.items()}


def analyze_sentiment(notes):
    notes_lower = notes.lower()
    pos = sum(1 for w in POSITIVE_KEYWORDS if w in notes_lower)
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in notes_lower)
    sentiment = "positive" if pos > neg else "negative" if neg > pos else "mixed"
    return {"sentiment": sentiment, "positive_count": pos, "negative_count": neg}
