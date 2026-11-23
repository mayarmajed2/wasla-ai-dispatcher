"""High-Recall Egyptian Arabic Complaint & Food Safety Escalation Detector.

Optimized for high recall on critical complaints:
- Food safety concerns are ALWAYS escalated immediately as CRITICAL, regardless of classifier confidence.
- Detects late delivery, missing items, wrong products, incorrect billing, food quality,
  cold food, poor service, delivery issues, refund requests, and frustrated/aggressive customers.
- Never promises unauthorized refunds or specific compensation deadlines.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ai_service.understanding.menu_index import normalize_ar


@dataclass(frozen=True)
class ComplaintAssessment:
    is_complaint: bool
    category: str
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW
    requires_owner_escalation: bool
    summary_en: str
    customer_reply_ar: str
    matched_signals: list[str]


# Ordered by urgency (CRITICAL first)
_COMPLAINT_RULES: list[tuple[str, str, str, list[str]]] = [
    (
        "food_safety",
        "CRITICAL",
        "Customer reported a potential food safety / hygiene issue.",
        [
            "تسمم", "بطني وجعتني", "وجع بطن", "مغص", "ترجيع",
            "بايظ", "بايظه", "فاسد", "فاسده", "معفن", "محمض",
            "حشره", "صرصار", "شعره", "شعر في الاكل", "ريحه وحشه", "ريحته وحشه",
            "مش نضيف", "قذر", "نيئ", "نيء", "مش مستوي",
        ],
    ),
    (
        "missing_item",
        "HIGH",
        "Customer reported one or more missing items from the order.",
        [
            "ناقص", "ناقصه", "مجاش", "ماجاش", "منسي", "نسيتو", "نسيتوا",
            "موصليش", "مش موجود في الكيس", "ناقص ساندوتش", "ناقص بطاطس", "ناقص مشروب",
        ],
    ),
    (
        "wrong_item",
        "HIGH",
        "Customer received the wrong product or incorrect order.",
        [
            "اوردر غلط", "طلب غلط", "الاوردر غلط", "الطلب غلط", "مش ده اللي طلبته", "مش اللي طلبته",
            "جالي غير", "جالي غلط", "بعتولي غلط", "متلخبط",
        ],
    ),
    (
        "incorrect_billing",
        "HIGH",
        "Customer reported an overcharge or billing discrepancy.",
        [
            "الحساب غلط", "السعر غلط", "حساب زيادة", "حساب زياده", "دفعت اكتر", "حاسبني غلط",
            "فاتوره غلط", "نصب", "باقي الفلوس", "ماخدتش الباقي",
        ],
    ),
    (
        "refund_request",
        "HIGH",
        "Customer requested a refund or compensation.",
        [
            "عايز فلوسي", "عايزه فلوسي", "رجعولي فلوسي", "ارجاع", "استرجاع", "ريفند", "تعويض",
        ],
    ),
    (
        "late_delivery",
        "HIGH",
        "Customer reported a significant delivery delay.",
        [
            "اتأخر", "متأخر", "متاخر", "اتأخرتوا", "بقالي ساعه", "بقالي ساعتين",
            "لسه موصلش", "فين الاوردر", "الأوردر اتأخر", "الطلب اتأخر", "تأخير", "تاخير",
        ],
    ),
    (
        "cold_food",
        "MEDIUM",
        "Customer reported cold food on arrival.",
        [
            "بارد", "بارده", "ساقع", "ساقعه", "مثلج", "مش سخن", "مش سخنه",
        ],
    ),
    (
        "food_quality",
        "HIGH",
        "Customer complained about poor food taste or quality.",
        [
            "وحش", "وحشه", "طعمه وحش", "طعمه سيء", "سيء", "سيئ", "محروق", "محروقه",
            "مالح اوي", "مالح جدا", "ناشف", "زفر", "زفاره", "جوده وحشه", "مش حلو خالص",
        ],
    ),
    (
        "delivery_problem",
        "HIGH",
        "Customer reported an issue with the delivery rider or drop-off.",
        [
            "الطيار", "المندوب", "الدليفري قليل الادب", "مندوب التوصيل", "مردش", "وقع الاكل", "الاكل اتقلب", "مقفل",
        ],
    ),
    (
        "aggressive_customer",
        "HIGH",
        "Frustrated or escalated customer requiring immediate human intervention.",
        [
            "حسبي الله", "ايه القرف ده", "قرف", "زفت", "اسوأ مطعم", "اسوا مطعم",
            "هعمل شكوى", "حمايه المستهلك", "مسخره", "فاشل", "اخر مره اطلب",
            "مش هطلب تاني", "مقرف", "خدمه زباله", "زباله",
        ],
    ),
    (
        "service_issue",
        "MEDIUM",
        "Customer reported a service complaint.",
        [
            "شكوى", "شكوي", "بشتكي", "عندي مشكله", "مشكله في الاوردر", "مشكله في الطلب",
            "معامله وحشه", "المدير", "صاحب المطعم", "المسؤول",
        ],
    ),
]

POLITE_COMPLAINT_REPLY_AR = (
    "متأسفين جداً على اللي حصل ❤️ "
    "وصلت شكوتك للمسؤول فوراً وهنتابعها معاك حالاً."
)

CRITICAL_FOOD_SAFETY_REPLY_AR = (
    "متأسفين جداً جداً على اللي حصل يا فندم، سلامتك أهم حاجة عندنا ❤️ "
    "تم تصعيد الشكوى فوراً وبأعلى أولوية لصاحب المطعم وهيتواصل معاك حالاً."
)


def assess_complaint(text: str, *, predicted_intent: str = "") -> ComplaintAssessment:
    """Detect and classify restaurant complaints with high recall on safety/critical issues."""
    norm = normalize_ar(text)
    matched: list[str] = []
    chosen_category = ""
    chosen_severity = "LOW"
    chosen_summary = ""

    for category, severity, summary, keywords in _COMPLAINT_RULES:
        hits = [kw for kw in keywords if normalize_ar(kw) in norm]
        if hits:
            matched.extend(hits)
            if not chosen_category:
                chosen_category = category
                chosen_severity = severity
                chosen_summary = summary

    # Guard against false positives on normal polite inquiries like "فين التوصيل" vs actual complaint
    if not chosen_category and predicted_intent == "complaint":
        chosen_category = "service_issue"
        chosen_severity = "MEDIUM"
        chosen_summary = "Customer submitted a complaint detected by intent classifier."
        matched.append("intent:complaint")

    # Negative tone heuristic for English/mixed frustrated messages
    if not chosen_category and re.search(r"\b(complaint|refund|missing|cold|late|wrong order|horrible|disgusting)\b", text, re.I):
        chosen_category = "service_issue"
        chosen_severity = "HIGH"
        chosen_summary = "Customer reported an order issue in English."
        matched.append("en_complaint_signal")

    if not chosen_category:
        return ComplaintAssessment(
            is_complaint=False,
            category="none",
            severity="LOW",
            requires_owner_escalation=False,
            summary_en="",
            customer_reply_ar="",
            matched_signals=[],
        )

    reply = CRITICAL_FOOD_SAFETY_REPLY_AR if chosen_severity == "CRITICAL" else POLITE_COMPLAINT_REPLY_AR
    return ComplaintAssessment(
        is_complaint=True,
        category=chosen_category,
        severity=chosen_severity,
        requires_owner_escalation=True,
        summary_en=chosen_summary,
        customer_reply_ar=reply,
        matched_signals=matched,
    )
