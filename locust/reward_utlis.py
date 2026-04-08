from typing import Dict, Tuple, Optional, List
# =========================
# Utilities
# =========================
def _normalize_text(s: str) -> str:
    """Normalize text by converting to lowercase and removing extra whitespace."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def extract_tag_content(text: str, tag: str) -> str:
    """Get the inner text of the last occurrence of <tag>...</tag> (case-insensitive)."""
    if not text:
        return ""
    matches = re.findall(fr"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    return matches[-1].strip() if matches else ""


def extract_prediction(text: str) -> str:
    """Extract classification label from model output."""
    t = text or ""

    # 1) <class> tag
    matches = re.findall(r"<class>\s*([^<]+?)\s*</class>", t, flags=re.IGNORECASE)
    if matches:
        candidate = _normalize_text(matches[-1])
        if candidate in ("0", "1"):
            return candidate
        if candidate in ALIASES_TO_ID:
            return str(ALIASES_TO_ID[candidate])

    # 2) Loose matching
    t_norm = _normalize_text(t)
    m_digit = re.search(r"\b([01])\b", t_norm)
    if m_digit:
        return m_digit.group(1)

    pos_hits = any(k in t_norm for k in ("present", "positive", "yes", "爆发", "发生", "有蝗虫", "有"))
    neg_hits = any(k in t_norm for k in ("absent", "negative", "no", "不爆发", "不发生", "无蝗虫", "无", "没有"))
    if pos_hits and not neg_hits:
        return "1"
    if neg_hits and not pos_hits:
        return "0"
    return "None"
