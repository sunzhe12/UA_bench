"""Single-letter classification parsing with ASCII case tolerance.

Model responses must contain exactly one displayed option letter (ASCII A-Z or a-z).
Aliases are used only to decode trusted option text, never model prose.
Invalid responses produce unknown labels and parse_status=invalid_format.
"""

import re
from typing import Dict, List, Tuple

from .scorer import HierPrediction


# ============================================================
# Option parsing - expand a short option identifier emitted by the model back into the full text
# ============================================================

def resolve_short_answer(output: str, prompt: str) -> str:
    """Resolve one ASCII option letter, ignoring case; reject other response forms.

    Surrounding whitespace and ASCII letter case are ignored. An empty result means invalid
    format or an option that is absent from the supplied question.
    """
    answer = output.strip()
    if not re.fullmatch(r"[A-Za-z]", answer):
        return ""
    options = dict(re.findall(r"^\s*([A-Z])\. +([^\r\n]+)", prompt, re.MULTILINE))
    return options.get(answer.upper(), "")


class OutputParser:
    """Parser for hierarchical classification outputs."""

    def __init__(self, config: dict):
        tax = config["taxonomy"]
        self.l1_map = self._build_alias_map(tax["L1"]["classes"])
        self.l2_map = self._build_alias_map(tax["L2"]["classes"])
        self.l3_map = self._build_alias_map(tax["L3"]["classes"])

    @staticmethod
    def _build_alias_map(classes: List[dict]) -> List[Tuple[str, str]]:
        """Build the (alias, key) list ordered by descending alias length (longest match first)."""
        pairs = []
        for c in classes:
            for alias in c["aliases"]:
                pairs.append((alias, c["key"]))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        return pairs

    # ============================================================
    # Main parsing entry point
    # ============================================================

    def parse(self, sample_id: str, model_output: str,
              prompt: str = "") -> HierPrediction:
        """Extract L1/L2/L3 from the model output.

        Args:
            sample_id: sample identifier
            model_output: raw model output
            prompt: the prompt sent to the model (used to resolve option letters)
        """
        # Option parsing: if the model emits only a short identifier such as "A", expand it to the full text
        original_output = model_output
        text = resolve_short_answer(model_output, prompt)

        l1, l1_tier = self._extract(text, self.l1_map)
        l2, l2_tier = self._extract(text, self.l2_map)
        l3, l3_tier = self._extract(text, self.l3_map)

        parse_tier = max(l1_tier, l2_tier, l3_tier)

        return HierPrediction(
            sample_id=sample_id,
            L1=l1, L2=l2, L3=l3,
            parse_tier=parse_tier,
            raw_output=original_output,
            parse_status="valid_option" if text else "invalid_format",
        )

    def parse_turn1(self, sample_id: str, output: str,
                    prompt: str = "") -> HierPrediction:
        """Turn 1 output - extract L1 only (skip the L2/L3 extraction work)."""
        original_output = output
        text = resolve_short_answer(output, prompt)
        l1, l1_tier = self._extract(text, self.l1_map)
        return HierPrediction(
            sample_id=sample_id,
            L1=l1, L2="unknown", L3="unknown",
            parse_tier=l1_tier,
            raw_output=original_output,
            parse_status="valid_option" if text else "invalid_format",
        )

    # ============================================================
    # Core matching
    # ============================================================

    def _extract(
        self, text: str, alias_map: List[Tuple[str, str]]
    ) -> Tuple[str, int]:
        """Two-round fallback matching -> (class_key, parse_tier)."""
        text_lower = text.lower()

        # Tier 1: direct alias match
        for alias, key in alias_map:
            if alias.lower() in text_lower:
                return key, 1

        # Tier 2: relaxed match after stripping non-alphanumeric characters
        text_clean = re.sub(r'[^\w]', '', text_lower)
        for alias, key in alias_map:
            alias_clean = re.sub(r'[^\w]', '', alias.lower())
            if alias_clean and alias_clean in text_clean:
                return key, 2

        return "unknown", 3


# ============================================================
# Ground-truth extraction (from JSON metadata)
# ============================================================

def extract_l1_l2_l3_from_meta(meta: dict) -> Dict[str, str]:
    """
    Extract the L1/L2/L3 ground truth from JSON metadata.

    PulseCom:
      signal_category = "pulse" → L1="active", L2="pulse"
      signal_category = "communication" → L1="active", L2="communication"
      signal_type = "CW"/"LFM"/... → L3
    Ship:
      signal_category = "radiated_noise" → L1="passive", L2="ship_noise"
      sub_type = "cargo"/"cruise"/... → L3
    """
    cat = meta.get("signal_category", "")

    if cat in ("pulse", "communication"):
        l1 = "active"
        l2 = "pulse" if cat == "pulse" else "communication"
    elif cat == "radiated_noise":
        l1 = "passive"
        l2 = "ship_noise"
    else:
        return {"L1": "unknown", "L2": "unknown", "L3": "unknown"}

    # L3: prefer sub_type, then signal_type, finally signal_params.original_class_name
    l3 = (meta.get("sub_type") or meta.get("signal_type")
          or (meta.get("signal_params") or {}).get("original_class_name", "unknown"))
    l3 = str(l3).lower().strip()

    # ship class name -> L3 key mapping
    ship_l3_map = {
        "cargo_ship": "cargo", "cruise_ship": "cruise",
        "fishing_boat": "fishing", "warship": "warship",
        "underwater_target": "underwater_target",
    }
    l3 = ship_l3_map.get(l3, l3)

    # PulseCom L3 normalisation (already CW/LFM/2FSK and so on, just upper-case it)
    if cat in ("pulse", "communication"):
        l3 = l3.upper()

    return {"L1": l1, "L2": l2, "L3": l3}
