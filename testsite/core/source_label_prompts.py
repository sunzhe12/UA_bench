"""Source-based L1 wording shared by the two QA generators.

Active denotes deliberate detection/communication transmission. Passive denotes
incidental noise radiated by an operating vessel or underwater vehicle, not a
receiver mode. Stable evaluation keys remain active/passive.
"""

QA_PROMPT_VERSION = "source_origin_v4_best_match_letter_only"
OPTION_ONLY_INSTRUCTION = "Select the option that you consider the best match for the audio. You must choose exactly one of the listed options. Reply with the uppercase option letter only, without explanation or any additional text."

T1_TEMPLATES = [
    "Determine whether this underwater sound is a deliberately transmitted signal or source-radiated noise.\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
    "Is the sound intentionally transmitted for detection or communication, or radiated as noise by an operating platform?\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
    "Classify the sound by its source: deliberate transmission or platform-radiated noise.\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
    "Does this audio contain an intentionally transmitted waveform or noise radiated by a vessel or underwater vehicle?\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
    "Identify the source category of this sound: a deliberate acoustic transmission or incidental platform noise.\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
    "Is this sound produced by deliberate signal transmission or by the operation of a vessel or underwater vehicle?\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
    "Distinguish a deliberately transmitted acoustic signal from noise radiated by an operating platform.\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
    "Which source category best describes this underwater sound: intentional transmission or source-radiated noise?\nOptions:\nA. Active - Deliberately transmitted signal\nB. Passive - Source-radiated noise",
]
T1_TEMPLATES = [q + "\n" + OPTION_ONLY_INSTRUCTION for q in T1_TEMPLATES]


def get_t1_answer(l1: str) -> str:
    """Return an explicit source-category option, rejecting unknown labels."""
    key = l1.strip().lower()
    if key in ("active", "actively transmitted", "deliberately transmitted"):
        return "A"
    if key in ("passive", "source-radiated", "source-radiated noise"):
        return "B"
    raise ValueError(f"unsupported L1 label: {l1!r}")
