"""
TitleForge AI — Text Pre/Post-Processing Utilities
"""
import re
import html
from typing import Optional

# ── Abbreviation expansion map ─────────────────────────────────────────────────
ABBREV_MAP = {
    r"\bblk\b":   "Black",
    r"\bwht\b":   "White",
    r"\bgry\b":   "Grey",
    r"\bslvr\b":  "Silver",
    r"\bnvy\b":   "Navy",
    r"\bred\b":   "Red",
    r"\bgrn\b":   "Green",
    r"\bpur\b":   "Purple",
    r"\bwrls\b":  "Wireless",
    r"\bwrl\b":   "Wireless",
    r"\bbt\b":    "Bluetooth",
    r"\bnoisecancelling\b": "Noise-Cancelling",
    r"\bnc\b":    "Noise-Cancelling",
    r"\bhdphn\b": "Headphones",
    r"\bhdph\b":  "Headphones",
    r"\bspkr\b":  "Speaker",
    r"\bportbl\b":"Portable",
    r"\bportb\b": "Portable",
    r"\bkb\b":    "Keyboard",
    r"\bmech\b":  "Mechanical",
    r"\bmoni\b":  "Monitor",
    r"\bmon\b":   "Monitor",
    r"\bcam\b":   "Camera",
    r"\bmach\b":  "Machine",
    r"\bmkr\b":   "Maker",
    r"\brcvr\b":  "Receiver",
    r"\bbattry\b":"Battery",
    r"\bbatty\b": "Battery",
    r"\bpwr\b":   "Power",
    r"\bstn\b":   "Station",
    r"\bsolar gen\b": "Solar Generator",
    r"\bssd\b":   "SSD",
    r"\bhdd\b":   "HDD",
    r"\bntwrk\b": "Network",
    r"\bwifi\b":  "Wi-Fi",
    r"\bwifi6\b": "Wi-Fi 6",
    r"\bwrks\b":  "Works",
    r"\bw/\b":    "with",
    r"\bw\b":     "with",
    r"\bgen\b":   "Generation",
    r"\bext\b":   "External",
    r"\bintrnl\b":"Internal",
    r"\bdual\b":  "Dual",
    r"\bpltfm\b": "Platform",
    r"\bpltnm\b": "Platinum",
    r"\bprgmbl\b":"Programmable",
    r"\bprogrammbl\b": "Programmable",
    r"\badj\b":   "Adjustable",
    r"\bsz(\d+)\b": r"Size \1",
    r"\bsz\b":    "Size",
    r"\babslt\b": "Absolute",
    r"\bclnr\b":  "Cleaner",
    r"\bvacm\b":  "Vacuum",
    r"\b(\d+)in\b": r"\1-Inch",
    r"\b(\d+)inch\b": r"\1-Inch",
    r"\bwreless\b": "Wireless",
    r"\bpodcst\b": "Podcast",
    r"\bpodcast\b": "Podcast",
    r"\bprofessional\b": "Professional",
    r"\bultralght\b": "Ultra-Lightweight",
    r"\bcompact\b": "Compact",
    r"\btwst\b":   "True Wireless",
    r"\bgph\b":    "Graphite",
    r"\bmon\b":    "Monitor",
    r"\brcvr\b":   "Receiver",
    r"\blrg\b":    "Large",
    r"\bmed\b":    "Medium",
    r"\bsml\b":    "Small",
}

# ── Noise patterns to strip ────────────────────────────────────────────────────
NOISE_PATTERNS = [
    r"!!+[^!]*!!+",          # !!SALE!!, !!HOT!!
    r"\*{2,}[^*]*\*{2,}",   # **SALE**, ***HOT***
    r"【[^】]*】",            # 【SALE】
    r"\[[^\]]*sale[^\]]*\]", # [SALE]
    r"<[^>]+>",              # HTML tags
    r"&\w+;",                # HTML entities (handled separately)
    r"\s{2,}",               # Multiple spaces (normalize to single)
]


def preprocess(title: str) -> str:
    """Clean raw title before passing to the model."""
    if not title:
        return ""

    # Decode HTML entities
    title = html.unescape(title)

    # Strip HTML tags
    title = re.sub(r"<[^>]+>", " ", title)

    # Remove noise patterns
    for pattern in NOISE_PATTERNS[2:]:  # Skip HTML ones already done
        title = re.sub(pattern, " ", title, flags=re.IGNORECASE)

    # Normalize whitespace
    title = re.sub(r"\s+", " ", title).strip()

    return title


def postprocess(title: str) -> str:
    """Clean and standardize model output."""
    if not title:
        return ""

    # Take only the first line (model sometimes adds explanation)
    title = title.split("\n")[0].strip()

    # Strip leading/trailing quotes
    title = title.strip('"\'')

    # Normalize whitespace
    title = re.sub(r"\s+", " ", title).strip()

    # Fix common punctuation issues
    title = re.sub(r"\s([,.:;])", r"\1", title)     # space before punct
    title = re.sub(r"([,(])\s+", r"\1", title)       # space after open paren/comma
    title = re.sub(r"\(\s+", "(", title)
    title = re.sub(r"\s+\)", ")", title)

    return title


def rule_based_normalize(title: str) -> str:
    """
    Lightweight rule-based fallback normalizer.
    Used when the AI model is not available or for pre-processing.
    """
    title = preprocess(title)

    # Remove promotional noise patterns
    for pattern in NOISE_PATTERNS[:2]:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)

    # Expand abbreviations
    for pattern, replacement in ABBREV_MAP.items():
        title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)

    # Apply Title Case
    title = title.title()

    # Fix known Title Case mistakes (acronyms, model numbers)
    title = re.sub(r"\bSsd\b",  "SSD",   title)
    title = re.sub(r"\bHdd\b",  "HDD",   title)
    title = re.sub(r"\bRam\b",  "RAM",   title)
    title = re.sub(r"\bUsb\b",  "USB",   title)
    title = re.sub(r"\bAmd\b",  "AMD",   title)
    title = re.sub(r"\bCpu\b",  "CPU",   title)
    title = re.sub(r"\bGpu\b",  "GPU",   title)
    title = re.sub(r"\bWi-Fi\b","Wi-Fi", title)
    title = re.sub(r"\bNvme\b", "NVMe",  title)
    title = re.sub(r"\bOled\b", "OLED",  title)
    title = re.sub(r"\bQled\b", "QLED",  title)
    title = re.sub(r"\bUhd\b",  "UHD",   title)
    title = re.sub(r"\bHdr\b",  "HDR",   title)
    title = re.sub(r"\bFhd\b",  "FHD",   title)
    title = re.sub(r"\b4K\b",   "4K",    title)
    title = re.sub(r"\b8K\b",   "8K",    title)
    title = re.sub(r"\bAtx\b",  "ATX",   title)

    # Normalize whitespace
    title = re.sub(r"\s+", " ", title).strip()

    return postprocess(title)


def truncate_for_display(text: str, max_len: int = 120) -> str:
    """Truncate a string for UI display."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"
