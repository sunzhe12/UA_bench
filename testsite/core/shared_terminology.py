"""
Shared L1/L2/L3 terminology for underwater acoustic classification.

This is the SINGLE SOURCE OF TRUTH for:
  - L3_PRECISE:  pipeline QA generation terms (archive/pipeline_step2_qa.py,
                 archive/pipeline_ship_step2_qa.py)
  - L3_SHOULD:   scorer evaluation terms (testsite/core/scorer.py)
  - L1/L2 descriptive terms for T3 reasoning generation

When terminology changes, update ONLY this file — all consumers re-import.

Two-tier design:
  - L3_PRECISE:  accurate observable concepts for each class
  - L3_SHOULD:   expanded synonym set used by the reasoning scorer

SFT rationales are built by build_l3_evidence(), which conditions directional
and ship-noise descriptions on each sample's saved metadata.
"""

import math

# ============================================================
# L1 descriptive terms — participle/adjective form for T3 templates
# ============================================================
L1_ACTIVE_TERMS = ["actively transmitted", "deliberately emitted", "purposefully transmitted"]
L1_PASSIVE_TERMS = ["source-radiated", "target-radiated", "self-radiated by the source platform"]

# ============================================================
# L2 descriptive terms — noun phrases keyed by L2 label
# ============================================================
L2_ACTIVE_TERMS = {
    "detection pulse":      ["detection pulse characteristics", "pulse detection mode",
                             "sonar pulse properties"],
    "communication signal": ["underwater acoustic communication features",
                             "digital communication modulation",
                             "communication modulation scheme"],
}
L2_PASSIVE_TERMS = {
    "ship-radiated noise": ["ship-radiated noise characteristics",
                            "vessel-radiated noise properties",
                            "target-radiated noise pattern"],
}

# ============================================================
# L3 PRECISE — for QA generation (SFT training data)
# Strictly observable waveform or spectral concepts. Directional and ship
# evidence is refined per sample by build_l3_evidence().
# ============================================================
L3_PRECISE = {
    # ── Detection pulse ──
    "CW": ["stable single-frequency tone", "unchanging frequency", "narrowband structure"],
    "LFM": ["linear frequency modulation", "straight time-frequency trajectory",
            "constant sweep rate"],
    "HFM": ["hyperbolic frequency modulation", "curved time-frequency trajectory",
            "time-varying sweep rate"],
    # ── Communication ──
    "2FSK": ["binary frequency-shift keying", "two discrete frequency states",
             "symbol-wise frequency switching"],
    "4FSK": ["quaternary frequency-shift keying", "four discrete frequency states",
             "symbol-wise frequency switching"],
    "BPSK": ["binary phase-shift keying", "two phase states",
             "approximately 180-degree phase transitions"],
    "QPSK": ["quadrature phase-shift keying", "four phase states",
             "symbol-wise phase-state transitions"],
    "OFDM": ["orthogonal frequency-division multiplexing", "simultaneous multicarrier structure",
             "regularly spaced subcarriers"],
    # ── Ship-radiated noise ──
    "cargo": ["broadband-plus-tonal vessel noise", "discrete machinery tonal components",
              "periodic propeller-related modulation"],
    "cruise": ["broadband-plus-tonal vessel noise", "discrete machinery tonal components",
               "periodic propeller-related modulation"],
    "fishing": ["broadband-plus-tonal vessel noise", "discrete machinery tonal components",
                "periodic propeller-related modulation"],
    "warship": ["broadband-plus-tonal vessel noise", "discrete machinery tonal components",
                "periodic propeller-related modulation"],
    "underwater_target": ["low-frequency tonal structure", "discrete machinery tonal components",
                          "weak periodic propeller-related modulation"],
}

# ============================================================
# L3 SHOULD — for scorer reasoning-quality evaluation
# Superset of L3_PRECISE: curated English variants and compatible shared cues.
# This finite vocabulary is not an exhaustive semantic description of a class.
# ============================================================
L3_SHOULD = {
    # ── Detection pulse ──
    "CW": {"stable single-frequency tone", "unchanging frequency", "narrowband structure",
           "single-frequency", "continuous wave", "single tone", "narrowband",
           "steady frequency", "unmodulated", "fixed frequency", "constant frequency"},
    "LFM": {"linear frequency modulation", "straight time-frequency trajectory",
            "constant sweep rate", "linear chirp", "linear sweep", "linear upsweep",
            "linear downsweep", "frequency increases linearly with time",
            "frequency decreases linearly with time", "frequency ramp", "chirp"},
    "HFM": {"hyperbolic frequency modulation", "curved time-frequency trajectory",
            "time-varying sweep rate", "hyperbolic chirp", "hyperbolic sweep",
            "nonlinear sweep", "nonlinear upsweep", "nonlinear downsweep",
            "frequency increases nonlinearly with time",
            "frequency decreases nonlinearly with time", "nonlinear chirp"},
    # ── Communication ──
    "2FSK": {"binary frequency-shift keying", "two discrete frequency states",
             "symbol-wise frequency switching", "binary FSK", "two frequency states",
             "frequency switching"},
    "4FSK": {"quaternary frequency-shift keying", "four discrete frequency states",
             "symbol-wise frequency switching", "quaternary FSK", "four frequency states",
             "multi-level FSK", "frequency switching"},
    "BPSK": {"binary phase-shift keying", "two phase states",
             "approximately 180-degree phase transitions", "phase reversal",
             "binary PSK", "phase flip", "phase inversion", "binary phase"},
    "QPSK": {"quadrature phase-shift keying", "four phase states",
             "symbol-wise phase-state transitions", "quadrature PSK",
             "four-phase", "quadrature modulation"},
    "OFDM": {"orthogonal frequency-division multiplexing", "simultaneous multicarrier structure",
             "regularly spaced subcarriers", "subcarriers", "multi-carrier",
             "orthogonal carriers", "multiple subcarriers", "dense carriers"},
}


_SHIP_OBSERVABLE_TERMS = {
    "broadband-plus-tonal vessel noise", "discrete machinery tonal components",
    "periodic propeller-related modulation", "low-frequency tonal structure",
    "weak periodic propeller-related modulation",
    "tonal components concentrated in the low-frequency band",
    "tonal components spread across the low-to-mid-frequency band",
    "tonal components spanning a broad frequency range",
    "sparse discrete tonal structure", "moderate-density discrete tonal structure",
    "dense discrete tonal structure", "shaft-rate-related harmonic lines",
    "blade-rate-related harmonic lines", "shaft- and blade-rate-related harmonic lines",
    "mechanical tonal components without a dominant harmonic family",
    "weak periodic amplitude modulation", "moderate periodic amplitude modulation",
    "strong periodic amplitude modulation",
}

for _ship_key, _label_term in {
    "cargo": "cargo vessel",
    "cruise": "cruise ship",
    "fishing": "fishing vessel",
    "warship": "naval vessel",
    "underwater_target": "underwater vehicle",
}.items():
    L3_SHOULD[_ship_key] = set(_SHIP_OBSERVABLE_TERMS) | {_label_term}


# Non-exclusive observations are shared explicitly. Membership means that the
# phrase is compatible with the class, not that every sample exhibits it or
# that it distinguishes the leaf. Keep state counts and modulation laws specific.
_SHARED_CUE_GROUPS = (
    (("LFM", "HFM"), {"chirp"}),
    (("BPSK", "QPSK"), {
        "symbol-wise phase-state transitions", "phase reversal", "phase flip",
        "phase inversion", "approximately 180-degree phase transitions",
    }),
    # Relative bandwidth / narrowband components alone do not specify a class.
    # Apply to all leaves so these generic phrases cannot imply a CW conflict.
    (tuple(L3_SHOULD), {"narrowband", "narrowband structure"}),
)
for _cue_classes, _cue_terms in _SHARED_CUE_GROUPS:
    for _cue_class in _cue_classes:
        L3_SHOULD[_cue_class].update(_cue_terms)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _directional_term(l3: str, direction: str, rng) -> str:
    direction = str(direction or "").strip().lower()
    if direction in {"upsweep", "up", "ascending"}:
        choices = {
            "LFM": ("linear upsweep", "frequency increases linearly with time"),
            "HFM": ("nonlinear upsweep", "frequency increases nonlinearly with time"),
        }
    elif direction in {"downsweep", "down", "descending"}:
        choices = {
            "LFM": ("linear downsweep", "frequency decreases linearly with time"),
            "HFM": ("nonlinear downsweep", "frequency decreases nonlinearly with time"),
        }
    else:
        choices = {
            "LFM": ("linear sweep", "linear chirp"),
            "HFM": ("nonlinear sweep", "hyperbolic chirp"),
        }
    return rng.choice(choices[l3])


def _ship_evidence(meta: dict, rng) -> list[str]:
    params = meta.get("signal_params") or {}
    if not isinstance(params, dict):
        params = {}
    lines = params.get("lines") or []
    if not isinstance(lines, list):
        lines = []

    low = high = None
    frequencies = [
        value for line in lines if isinstance(line, dict)
        if (value := _number(line.get("frequency_hz"))) is not None
    ]
    if frequencies:
        low, high = min(frequencies), max(frequencies)
    else:
        frequency_range = params.get("line_frequency_range_hz")
        if isinstance(frequency_range, (list, tuple)) and len(frequency_range) >= 2:
            low, high = _number(frequency_range[0]), _number(frequency_range[1])
    if high is None:
        band_term = "broadband-plus-tonal vessel noise"
    elif high <= 250:
        band_term = "tonal components concentrated in the low-frequency band"
    elif high is not None and (low is None or low < 250) and high <= 1200:
        band_term = "tonal components spread across the low-to-mid-frequency band"
    else:
        band_term = "tonal components spanning a broad frequency range"

    count_value = _number(params.get("num_lines"))
    count = int(count_value) if count_value is not None and count_value >= 0 else len(lines)
    if count_value is None and not lines:
        density_term = "discrete machinery tonal components"
    elif count <= 4:
        density_term = "sparse discrete tonal structure"
    elif count <= 7:
        density_term = "moderate-density discrete tonal structure"
    else:
        density_term = "dense discrete tonal structure"

    source_types = {
        str(line.get("source_type", "")).strip().lower()
        for line in lines if isinstance(line, dict)
    }
    has_shaft = "shaft_harmonic" in source_types
    has_blade = "blade_harmonic" in source_types
    if has_shaft and has_blade:
        harmonic_term = "shaft- and blade-rate-related harmonic lines"
    elif has_shaft:
        harmonic_term = "shaft-rate-related harmonic lines"
    elif has_blade:
        harmonic_term = "blade-rate-related harmonic lines"
    elif source_types:
        harmonic_term = "mechanical tonal components without a dominant harmonic family"
    else:
        harmonic_term = "periodic propeller-related modulation"

    coefficients = [
        value for key in ("shaft_mod_coeff", "blade_mod_coeff")
        if (value := _number(params.get(key))) is not None
    ]
    strength = max(coefficients) if coefficients else None
    if strength is not None and strength <= 0.09:
        modulation_term = "weak periodic amplitude modulation"
    elif strength is not None and strength <= 0.14:
        modulation_term = "moderate periodic amplitude modulation"
    elif strength is not None:
        modulation_term = "strong periodic amplitude modulation"
    else:
        modulation_term = "periodic propeller-related modulation"

    # Keep both complementary propulsion cues because each is independently
    # supported by the saved line metadata and modulation coefficients.
    propulsion_term = rng.choice((
        f"{harmonic_term} with {modulation_term}",
        f"{modulation_term} associated with {harmonic_term}",
    ))
    return [band_term, density_term, propulsion_term]


def build_l3_evidence(l3: str, meta: dict, rng) -> list[str]:
    """Build reproducible, metadata-conditioned, observable L3 evidence."""
    if l3 not in L3_PRECISE:
        raise ValueError(f"unsupported L3 label: {l3!r}")
    if l3 in {"cargo", "cruise", "fishing", "warship", "underwater_target"}:
        return _ship_evidence(meta, rng)

    params = meta.get("signal_params") or {}
    if not isinstance(params, dict):
        params = {}
    if l3 in {"LFM", "HFM"}:
        return [
            L3_PRECISE[l3][0],
            L3_PRECISE[l3][1],
            _directional_term(l3, params.get("sweep_direction"), rng),
        ]
    return list(L3_PRECISE[l3])

# Auto-derived after shared cues are added: global scoring vocabulary
_ALL_L3_TERMS = set()
for _s in L3_SHOULD.values():
    _ALL_L3_TERMS.update(_s)


def get_should_not(l3: str) -> set:
    """Return terms that should NOT appear in reasoning for a given L3 class."""
    own = L3_SHOULD.get(l3, set())
    return _ALL_L3_TERMS - own
