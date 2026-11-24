"""Advanced Egyptian Arabic Normalization, Phonetic Folding, Franco-Arabic & Fuzzy Matching.

Designed specifically for real Egyptian restaurant customers typing on WhatsApp:
- Preserves original customer text in conversation history; normalization is used strictly for matching.
- Pure-Python Damerau-Levenshtein + token-alignment + phonetic skeleton (zero native DLL dependencies).
- Generalizes to unseen typos, missing/extra/repeated letters, and colloquial phrasing without hardcoding
  every possible misspelling.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Eastern Arabic digits -> Western digits
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Common Franco-Arabic & English restaurant vocabulary -> Arabic canonical tokens
FRANCO_TO_ARABIC: dict[str, str] = {
    "pizza": "بيتزا",
    "piza": "بيتزا",
    "pitza": "بيتزا",
    "margherita": "مارجريتا",
    "margarita": "مارجريتا",
    "margrita": "مارجريتا",
    "pepperoni": "بيبروني",
    "peperoni": "بيبروني",
    "burger": "برجر",
    "borger": "برجر",
    "classic": "كلاسيك",
    "cheese": "تشيز",
    "cheesy": "تشيزي",
    "chicken": "تشكن",
    "chiken": "تشكن",
    "crispy": "كريسبي",
    "crispi": "كريسبي",
    "crepe": "كريب",
    "crip": "كريب",
    "zinger": "زنجر",
    "ranch": "رانش",
    "bbq": "باربيكيو",
    "barbecue": "باربيكيو",
    "shawarma": "شاورما",
    "shish": "شيش",
    "pasta": "باستا",
    "negresco": "نجرسكو",
    "alfredo": "الفريدو",
    "fries": "بطاطس",
    "coleslaw": "كولسلو",
    "waffle": "وافل",
    "pancake": "بان كيك",
    "nutella": "نوتيلا",
    "lotus": "لوتس",
    "kinder": "كيندر",
    "pistachio": "بستاشيو",
    "oreo": "اوريو",
    "milkshake": "ميلك شيك",
    "latte": "لاتيه",
    "espresso": "اسبرسو",
    "cappuccino": "كابوتشينو",
    "tiramisu": "تيراميسو",
    "large": "كبير",
    "medium": "وسط",
    "single": "سنجل",
    "double": "دبل",
    "extra": "اكسترا",
    "sauce": "صوص",
    "no": "بدون",
    "3ayez": "",
    "3ayza": "",
    "3awez": "",
    "3awza": "",
    "wahed": "واحد",
    "wa7ed": "واحد",
    "etnen": "اتنين",
    "etnin": "اتنين",
    "talata": "تلاته",
    "w": "و",
}

# Egyptian colloquial request/politeness fillers stripped ONLY during entity extraction
EGYPTIAN_FILLERS = {
    "عايز", "عايزه", "عاوزه", "عاوز", "محتاج", "محتاجه", "نفسي", "في",
    "ممكن", "تجيبلي", "هات", "هاتلي", "اعملي", "اعمللي", "اطلب", "بطلب",
    "ابعتلي", "اديني", "جيبلي", "لو", "سمحت", "سمحتي", "بعد", "اذنك",
    "يا", "فندم", "باشا", "كابتن", "من", "فضلك", "بكام", "سعر", "سعره",
    "سعرها", "اسعار", "بكم", "شيل", "احذف", "امسح", "لغي", "الغي",
    "اقصد", "قصدي", "لا", "خليها", "خليه", "بدلها", "بدله", "ب", "علي",
    "واحد", "واحده", "اتنين", "اثنين", "تنين", "تلاته", "ثلاثه", "اربعه", "خمسه", "سته",
    "كبير", "كبيره", "لارج", "وسط", "ميديم", "سنجل", "دبل", "دابل", "صغير", "صغيره",
    "قطع", "قطعه", "حبه", "كامله", "حشو", "اطراف", "زياده", "اكسترا",
    "بقولك", "باقولك", "والله", "طيب", "طب", "انا", "عندكم", "عندكو", "فيه", "موجود", "فيها",
}

# Multi-word colloquial phrases to strip cleanly before entity extraction
FILLER_PHRASES = [
    r"لو\s+سمحت[ي]?",
    r"بعد\s+اذنك",
    r"من\s+فضلك",
    r"نفسي\s+في",
    r"عايز[ه]?\s+اطلب",
    r"عاوز[ه]?\s+اطلب",
    r"ممكن\s+تجيبلي",
    r"ممكن\s+اطلب",
    r"ممكن\s+واحد[ه]?",
    r"حبه\s+كامله",
    r"حشو\s+اطراف",
    r"بصل\s+مكرمل",
    r"صوص\s+من\s+اختيارك",
    r"اكسترا\s+بسطرمه",
    r"اكسترا\s+بيبروني",
    r"اكسترا\s+صوص",
    r"صوص\s+زياده",
    r"من\s+غير\s+[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)?",
    r"بدون\s+[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)?",
    r"(?:10|20)\s+قطع[ه]?",
]


def normalize_arabic(text: str) -> str:
    """Standard Egyptian Arabic normalization preserving meaningful menu distinctions."""
    if not text:
        return ""
    s = text.strip().lower().translate(_ARABIC_DIGITS)
    # Remove Tatweel and Arabic diacritics
    s = s.replace("\u0640", "")
    s = re.sub(r"[\u064B-\u065F\u0670]", "", s)
    # Normalize Alef, Yaa, Taa Marbuta, Hamza, Persian/Urdu variants
    s = re.sub(r"[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ی", "ي").replace("ک", "ك")
    s = s.replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    # Collapse 3+ repeated characters (e.g., بيتزاااا -> بيتزا)
    s = re.sub(r"(.)\1{2,}", r"\1", s)
    # Replace punctuation with spaces (keep Arabic, Latin, digits)
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Translate Franco-Arabic / English tokens inline
    tokens = []
    for tok in s.split():
        mapped = FRANCO_TO_ARABIC.get(tok, tok)
        if mapped:
            tokens.extend(mapped.split())
    return " ".join(tokens)


def phonetic_arabic(text: str) -> str:
    """Phonetic Egyptian Arabic folding for typo-tolerant matching ONLY."""
    s = normalize_arabic(text)
    if not s:
        return ""
    # Collapse 2+ repeated letters (e.g., ماارجريتا -> مارجريتا, بييتزا -> بيتزا)
    s = re.sub(r"(.)\1+", r"\1", s)
    # Common Egyptian loanword & dialect spelling equivalences
    replacements = [
        ("تشيكن", "تشكن"),
        ("تشكين", "تشكن"),
        ("تشيكن", "تشكن"),
        ("فراخ", "تشكن"),
        ("دجاج", "تشكن"),
        ("مارغريتا", "مارجريتا"),
        ("مرجريتا", "مارجريتا"),
        ("مارجرتا", "مارجريتا"),
        ("موتزاريلا", "موتزريلا"),
        ("موزاريلا", "موتزريلا"),
        ("موزريلا", "موتزريلا"),
        ("باربيكيو", "باربكيو"),
        ("بربكيو", "باربكيو"),
        ("هالابينو", "هالبينو"),
        ("ساندوتش", "سندوتش"),
        ("ساندويتش", "سندوتش"),
        ("شورما", "شاورما"),
        ("كوردون", "كوردن"),
        ("باسطرمه", "بسطرمه"),
        ("شوكليت", "شوكلت"),
        ("شيكولاته", "شوكلت"),
        ("شوكولاته", "شوكلت"),
        ("كرسبي", "كريسبي"),
        ("ستريس", "ستربس"),
        ("استربس", "ستربس"),
        ("ميكس", "مكس"),
        ("ميكش", "مكس"),
        ("تستي", "تيستي"),
        ("فرابتشينو", "فرابتشينو"),
        ("فرابوتشينو", "فرابتشينو"),
        ("بيبرونى", "بيبروني"),
        ("ببروني", "بيبروني"),
        ("ببيروني", "بيبروني"),
        ("كابوتشينو", "كابتشينو"),
        ("غ", "ج"),
    ]
    for src, dst in replacements:
        s = s.replace(src, dst)
    return re.sub(r"\s+", " ", s).strip()


def strip_fillers(text: str) -> str:
    """Remove Egyptian colloquial request fillers, quantities, sizes, and modifiers to isolate item text."""
    s = normalize_arabic(text)
    for pat in FILLER_PHRASES:
        s = re.sub(pat, " ", s)
    tokens = [t for t in s.split() if t not in EGYPTIAN_FILLERS and not t.isdigit()]
    return " ".join(tokens).strip()


def damerau_levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Damerau-Levenshtein similarity in [0.0, 1.0] (handles transpositions & typos)."""
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0
    max_len = max(len_a, len_b)
    # Fast length-difference bound
    if abs(len_a - len_b) / max_len > 0.65:
        return 0.0

    d = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        d[i][0] = i
    for j in range(len_b + 1):
        d[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition

    return 1.0 - (d[len_a][len_b] / max_len)


def hybrid_similarity(query: str, target: str) -> float:
    """Compute calibrated hybrid similarity between a customer query and a canonical MOKA item name.

    Combines:
    1. Phonetic-folded Damerau-Levenshtein similarity (great for single/double-char typos like مارجرينا/مرجريتا).
    2. SequenceMatcher ratio on normalized strings.
    3. Token-level best-pair alignment (handles reordered tokens or extra/missing category prefix).
    """
    q_norm = normalize_arabic(query)
    t_norm = normalize_arabic(target)
    if not q_norm or not t_norm:
        return 0.0
    if q_norm == t_norm:
        return 1.0

    q_ph = phonetic_arabic(q_norm)
    t_ph = phonetic_arabic(t_norm)
    if q_ph == t_ph:
        return 0.98

    # Character-level full string similarity
    dl_score = damerau_levenshtein_similarity(q_ph, t_ph)
    seq_score = SequenceMatcher(None, q_ph, t_ph).ratio()
    char_score = 0.65 * dl_score + 0.35 * seq_score

    # Token-level fuzzy alignment
    q_toks = q_ph.split()
    t_toks = t_ph.split()
    if q_toks and t_toks:
        q_to_t = sum(max(damerau_levenshtein_similarity(qt, tt) for tt in t_toks) for qt in q_toks) / len(q_toks)
        t_to_q = sum(max(damerau_levenshtein_similarity(tt, qt) for qt in q_toks) for tt in t_toks) / len(t_toks)
        # Asymmetric precision-weighted harmonic mean (penalizes hallucinating unrelated extra tokens)
        if q_to_t + t_to_q > 0:
            token_f1 = (2 * q_to_t * t_to_q) / (q_to_t + t_to_q)
        else:
            token_f1 = 0.0
        # Slight boost when query is a crisp subset of target (e.g., "مارجريتا" -> "بيتزا مارجريتا")
        subset_bonus = 0.88 * q_to_t if (len(q_toks) == 1 and len(t_toks) == 2 and q_to_t >= 0.80) else 0.0
    else:
        token_f1 = 0.0
        subset_bonus = 0.0

    return round(max(char_score, token_f1, subset_bonus), 4)
