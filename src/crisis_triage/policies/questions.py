"""The typed questions put to Decider 2B, and how a turn becomes `state`.

Wording is the main tuning lever for this model (docs/DECIDER.md section 1), so the
formulations here are not arbitrary -- `is_emergency` was chosen over three scored
alternatives ("immediate physical danger right now", "send 911 immediately", and a
`score` on imminence) because it was the only one that separated at all on the
fixture set. See docs/GROUND_TRUTH.md section 4 for the numbers.

No wording we tried separated *cleanly*, which is precisely why the band policy
cannot be the only thing standing between a user and crisis resources. These
questions drive tone and escalation; safety routing is backed by Bedrock.
"""

from __future__ import annotations

# Every question below is sent in ONE request so the server shares the state's KV
# prefix across them (strands_decider/infer.py:360).

IS_EMERGENCY = "is_emergency"
ESCALATING = "escalating"
CRISIS_TYPE = "crisis_type"
URGENCY_LEVEL = "urgency_level"

CRISIS_QUESTIONS: dict[str, dict] = {
    IS_EMERGENCY: {
        "type": "noul",
        "instructions": (
            "Does the latest message indicate a genuine emergency requiring "
            "immediate crisis resources?"
        ),
        "criteria": {
            "true": (
                "Imminent danger to self or others, an active suicide/violence plan, "
                "or a medical emergency requiring 911"
            ),
            "false": (
                "Distressed, anxious, venting, seeking support, or discussing past "
                "events, but not in immediate danger"
            ),
        },
    },
    ESCALATING: {
        "type": "noul",
        "instructions": (
            "Compared with the earlier messages, is the situation becoming more dangerous?"
        ),
        "criteria": {
            "true": (
                "Later messages show new intent, a plan, means, a timeline, or "
                "immediate danger not present earlier"
            ),
            "false": "Stable or improving; no new risk signals",
        },
    },
    CRISIS_TYPE: {
        "type": "choice",
        "instructions": "What type of crisis is this?",
        "criteria": {
            "suicide_self_harm": "Immediate risk of self-harm or suicide (plan, means, intent)",
            "domestic_violence": "Active domestic violence, abuse, or intimate partner violence",
            "medical_emergency": "Medical crisis requiring 911/ER (overdose, severe injury)",
            "mental_health_crisis": "Severe episode (psychosis, mania, unable to care for self)",
            "substance_overdose": "Suspected overdose, acute intoxication danger, or withdrawal emergency",
            "not_crisis": "Emotional distress but no immediate safety risk",
        },
    },
    URGENCY_LEVEL: {
        "type": "score",
        "instructions": "How urgent is the response needed?",
        # `score` criteria are ascending: index 0 is the low end (schema.py:73).
        "criteria": [
            "immediate: Seconds to minutes - life in danger right now",
            "urgent: Minutes to hours - safety at risk soon",
            "elevated: Needs professional help today, not life-threatening this moment",
            "routine: Can wait for a scheduled appointment or standard support",
        ],
    },
}

# The subset the band policy actually routes on. `crisis_type` and `urgency_level` are
# reported for context and the demo, not used for the routing decision.
ROUTING_QUESTIONS = (IS_EMERGENCY, ESCALATING)

DEFAULT_MAX_TURNS = 6


def build_state(user_turns: list[str], max_turns: int = DEFAULT_MAX_TURNS) -> str:
    """The state handed to Decider: the last N user turns, oldest first.

    Only user turns go in -- not the agent's replies or internal reasoning
    (docs/DECIDER.md section 1, rule 3). Turns are labelled so the `escalating`
    question has something to compare the latest message against.
    """
    recent = [turn.strip() for turn in user_turns if turn and turn.strip()][-max_turns:]
    if not recent:
        raise ValueError("need at least one non-empty user turn")
    if len(recent) == 1:
        return recent[0]
    return "\n".join(f"USER: {turn}" for turn in recent)