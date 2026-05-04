# ============================================================
# waste_analysis.py  –  Smart Waste Classification Core Logic
#
# KEY FIX: plastic_bag moved to non_recyclable (most recycling
# centres reject soft plastics). If your model labels differ,
# adjust CATEGORY_MAPPING to match your training classes.
# ============================================================

CATEGORY_MAPPING = {
    # ── Non-recyclable / hazardous ──────────────────────────
    "battery":                  "non_recyclable",
    "chemical_plastic_bottle":  "non_recyclable",
    "chemical_plastic_gallon":  "non_recyclable",
    "chemical_spray_can":       "non_recyclable",
    "light_bulb":               "non_recyclable",
    "paint_bucket":             "non_recyclable",
    "snack_bag":                "non_recyclable",
    "stick":                    "non_recyclable",
    "straw":                    "non_recyclable",
    "food_waste":               "non_recyclable",
    "plastic_cover":            "non_recyclable",   # ← your test item
    "dirty_item":               "non_recyclable",
    "plastic_bag":              "non_recyclable",
    "scrap_plastic":            "non_recyclable",   # FIXED: was incorrectly recyclable

    # ── Recyclable ──────────────────────────────────────────
    "can":                      "recyclable",
    "cardboard_bowl":           "recyclable",
    "cardboard_box":            "recyclable",
    "plastic_bottle":           "recyclable",
    "plastic_bottle_cap":       "recyclable",
    "plastic_box":              "recyclable",
    "plastic_cutlery":          "recyclable",
    "plastic_cup":              "recyclable",
    "plastic_cup_lid":          "recyclable",
    "reusable_paper":           "recyclable",
    "scrap_paper":              "recyclable",
    "paper":                    "recyclable",
    "metal_can":                "recyclable",
}

# Confidence threshold — detections below this are flagged but still counted
CONFIDENCE_THRESHOLD = 0.40


def analyze_waste(detected_classes, detected_confidences=None):
    """
    Analyse detected waste classes and return a full result dict.

    Parameters
    ----------
    detected_classes     : list[str]
    detected_confidences : list[float] | None
        Defaults to 1.0 — keeps test_model.py (1-arg call) working.

    Returns  (all keys listed for clarity)
    -------
    status, risk_level, contamination_percent,
    recyclable_count, non_recyclable_count,
    recyclable_items, non_recyclable_items, unknown_items,
    alert       → emoji string  (use in Streamlit)
    alert_text  → plain ASCII   (use in cv2.putText — NO ??? bug)
    recommendation, insight,
    recyclable, non_recyclable  (legacy aliases)
    """
    if detected_confidences is None:
        detected_confidences = [1.0] * len(detected_classes)

    recyclable_items     = []
    non_recyclable_items = []
    unknown_items        = []
    low_confidence_items = []

    for cls, conf in zip(detected_classes, detected_confidences):
        if conf < CONFIDENCE_THRESHOLD:
            low_confidence_items.append(cls)

        category = CATEGORY_MAPPING.get(cls.lower().strip(), "unknown")
        if category == "recyclable":
            recyclable_items.append(cls)
        elif category == "non_recyclable":
            non_recyclable_items.append(cls)
        else:
            unknown_items.append(cls)

    recyclable_count     = len(recyclable_items)
    non_recyclable_count = len(non_recyclable_items)
    total_items          = recyclable_count + non_recyclable_count
    low_confidence_flag  = len(low_confidence_items) > 0

    # Dynamic contamination % (never hardcoded)
    contamination_percent = (
        0 if total_items == 0
        else round((non_recyclable_count / total_items) * 100)
    )

    # Status + risk
    if total_items == 0:
        status, risk_level = "No Waste Detected", "None"
    elif recyclable_count > 0 and non_recyclable_count == 0:
        status, risk_level = "Clean Recyclable Waste", "Clean"
    elif recyclable_count > 0 and non_recyclable_count > 0:
        status = "Contaminated / Mixed Waste"
        risk_level = "High" if contamination_percent >= 50 else "Medium"
    elif recyclable_count == 0 and non_recyclable_count > 0:
        status, risk_level = "Non-Recyclable Waste", "Medium"
    else:
        status, risk_level = "Uncertain", "Low"

    # Two alert strings per status
    #   alert      → emoji, safe for Streamlit st.info / st.warning
    #   alert_text → plain ASCII, safe for cv2.putText (no ??? rendering bug)
    _data = {
        "Clean Recyclable Waste": (
            "Suitable for recycling.",
            "Dispose in the recycling bin.",
            "Proper recycling reduces landfill waste and conserves resources."
        ),
        "Contaminated / Mixed Waste": (
            "WARNING: Contaminated / mixed waste.",
            "Separate and clean recyclable items before disposal.",
            "Contaminated recyclables are often rejected at recycling plants."
        ),
        "Non-Recyclable Waste": (
            "ALERT: Non-recyclable item detected.",
            "Dispose in the general waste bin.",
            "Proper segregation prevents recycling stream contamination."
        ),
        "No Waste Detected": (
            "No waste detected.",
            "Point the camera at a waste item.",
            "Ensure good lighting and the item is fully visible."
        ),
    }
    _default = (
        "Detection uncertain.",
        "Reposition item under better lighting.",
        "Clear visibility improves AI detection accuracy."
    )
    alert_text, recommendation, insight = _data.get(status, _default)

    # emoji version for Streamlit
    _emoji = {
        "Clean Recyclable Waste":     "✅ " + alert_text,
        "Contaminated / Mixed Waste": "⚠️ " + alert_text,
        "Non-Recyclable Waste":       "❌ " + alert_text,
        "No Waste Detected":          "ℹ️ " + alert_text,
    }
    alert_emoji = _emoji.get(status, "⚠️ " + alert_text)

    if low_confidence_flag and total_items > 0:
        alert_emoji += " (Low confidence)"
        alert_text  += " (Low conf)"
        recommendation += " Improve lighting."

    return {
        "status":                status,
        "risk_level":            risk_level,
        "contamination_percent": contamination_percent,
        "recyclable_count":      recyclable_count,
        "non_recyclable_count":  non_recyclable_count,
        "recyclable_items":      recyclable_items,
        "non_recyclable_items":  non_recyclable_items,
        "unknown_items":         unknown_items,
        "low_confidence_items":  low_confidence_items,
        "alert":                 alert_emoji,   # Streamlit
        "alert_text":            alert_text,    # OpenCV — no emoji
        "recommendation":        recommendation,
        "insight":               insight,
        # legacy aliases so test_model.py doesn't break
        "recyclable":            recyclable_count,
        "non_recyclable":        non_recyclable_count,
    }