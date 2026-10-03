import os
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# ENVIRONMENT
# ============================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    ROOT / ".env"
)

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found in .env file."
    )


client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if str(value).lower() == "nan":
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def safe_text(value, default="Unknown"):
    if value is None:
        return default

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return default

    return text


# ============================================================
# PRAEVIA AI EXPLANATION
# ============================================================

def generate_explanation(lead):

    # --------------------------------------------------------
    # PRAEVIA'S EXISTING DECISION
    # --------------------------------------------------------

    priority = safe_text(
        lead.get("priority"),
        "LOW"
    )

    probability = safe_float(
        lead.get("conversion_probability")
    )

    decision_score = safe_float(
        lead.get("decision_score")
    )

    origin = safe_text(
        lead.get("origin"),
        "Unknown"
    )

    origin_rate = safe_float(
        lead.get("origin_rate")
    )

    origin_signal = safe_text(
        lead.get("origin_signal"),
        "N/A"
    )

    landing_page = safe_text(
        lead.get("landing_page_id"),
        "Unknown"
    )

    landing_rate = safe_float(
        lead.get("landing_rate")
    )

    landing_signal = safe_text(
        lead.get("landing_signal"),
        "N/A"
    )

    # --------------------------------------------------------
    # EXACT ACTION FROM DECISION ENGINE
    # --------------------------------------------------------

    action = safe_text(
        lead.get("next_best_action"),
        ""
    )

    if not action:
        action = safe_text(
            lead.get("recommended_action"),
            ""
        )

    if not action:
        action = "Review and follow up."

    # --------------------------------------------------------
    # STRUCTURED EVIDENCE
    # --------------------------------------------------------

    evidence = {
        "priority": priority,

        "conversion_probability":
            f"{probability * 100:.1f}%",

        "decision_score":
            f"{decision_score:.3f}",

        "origin": origin,

        "origin_historical_rate":
            f"{origin_rate * 100:.1f}%",

        "origin_signal":
            origin_signal,

        "landing_page":
            landing_page,

        "landing_historical_rate":
            f"{landing_rate * 100:.1f}%",

        "landing_signal":
            landing_signal,

        "exact_next_best_action":
            action,
    }

    # --------------------------------------------------------
    # GROUNDED PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the explanation layer of PRAEVIA,
an AI Decision Intelligence system for sales.

IMPORTANT:

PRAEVIA has ALREADY made the decision.

You are NOT the decision maker.

Your ONLY job is to explain the supplied decision
using the supplied evidence.

STRICT RULES:

1. Never change the priority.
2. Never change the conversion probability.
3. Never calculate a new score.
4. Never change the decision score.
5. Never invent evidence.
6. Never invent sample sizes.
7. Never mention sample sizes.
8. Never say that a sample size is zero.
9. Never invent facts about the lead.
10. Never invent a recommended action.
11. The NEXT ACTION must be copied EXACTLY.
12. If evidence is LIMITED, explicitly say that it should
    be interpreted cautiously.
13. Do not call the decision score a confidence score.
14. Do not say "immediate attention" unless that exact
    wording exists in the supplied action.
15. Keep the explanation concise and professional.

SUPPLIED PRAEVIA DECISION:

{evidence}

Return EXACTLY this structure:

WHY:
Write 1-2 concise sentences explaining why PRAEVIA
assigned the supplied priority.

EVIDENCE:
- State one factual evidence point.
- State one additional factual evidence point.

NEXT ACTION:
{action}

CONFIDENCE NOTE:
If either evidence signal is LIMITED, say:
"The evidence includes a LIMITED signal and should be
interpreted cautiously."

Otherwise say:
"The available evidence supports this recommendation."

Do not add any other sections.
"""

    # --------------------------------------------------------
    # OPENROUTER REQUEST
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model="openrouter/free",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grounded explanation layer. "
                    "Never override a business decision."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0.1,
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = response.choices[0].message.content

    if not result:
        return (
            "WHY:\n"
            "PRAEVIA generated a decision from the supplied "
            "conversion probability and evidence.\n\n"
            "EVIDENCE:\n"
            f"- Conversion probability: "
            f"{probability * 100:.1f}%.\n"
            f"- Origin signal: {origin_signal}.\n\n"
            "NEXT ACTION:\n"
            f"{action}\n\n"
            "CONFIDENCE NOTE:\n"
            "The available evidence should be reviewed by a human."
        )

    # --------------------------------------------------------
    # FINAL SAFETY CLEANUP
    # --------------------------------------------------------

    forbidden = [
        "sample size",
        "sample sizes",
        "sample-size",
        "sample-sizes",
        "zero sample",
        "zero samples",
    ]

    cleaned_lines = []

    for line in result.splitlines():

        lower_line = line.lower()

        if any(
            phrase in lower_line
            for phrase in forbidden
        ):
            continue

        cleaned_lines.append(line)

    result = "\n".join(
        cleaned_lines
    ).strip()

    # --------------------------------------------------------
    # GUARANTEE EXACT ACTION
    # --------------------------------------------------------

    # If the model accidentally changes the action,
    # replace the NEXT ACTION section with PRAEVIA's
    # canonical action.

    if "NEXT ACTION:" in result:

        before, after = result.split(
            "NEXT ACTION:",
            1
        )

        if "CONFIDENCE NOTE:" in after:

            _, confidence = after.split(
                "CONFIDENCE NOTE:",
                1
            )

            result = (
                before.strip()
                + "\n\nNEXT ACTION:\n"
                + action
                + "\n\nCONFIDENCE NOTE:\n"
                + confidence.strip()
            )

        else:

            result = (
                before.strip()
                + "\n\nNEXT ACTION:\n"
                + action
            )

    else:

        result += (
            "\n\nNEXT ACTION:\n"
            + action
        )

    return result