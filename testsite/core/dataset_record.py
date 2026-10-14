"""Self-contained dataset records shared by QA construction and evaluation."""
from copy import deepcopy
import math
from pathlib import Path
import re

RECORD_SCHEMA_VERSION = "ua_bench_record_v1"
LEAF_PARENTS = {
    **{key: ("active", "pulse") for key in ("CW", "LFM", "HFM")},
    **{key: ("active", "communication") for key in ("2FSK", "4FSK", "BPSK", "QPSK", "OFDM")},
    **{key: ("passive", "ship_noise") for key in ("cargo", "cruise", "fishing", "warship", "underwater_target")},
}


def validate_gt(value, sample_id):
    if not isinstance(value, dict) or not all(isinstance(value.get(k), str) for k in ("L1", "L2", "L3")):
        raise ValueError(f"{sample_id}: missing or incomplete embedded _gt; regenerate with the updated Step 2. Original metadata lookup is not supported.")
    parents = LEAF_PARENTS.get(value["L3"])
    if parents != (value["L1"], value["L2"]):
        raise ValueError(f"{sample_id}: unsupported or inconsistent embedded label path: {value}")
    return {key: value[key] for key in ("L1", "L2", "L3")}


def resolve_source_audio(meta, root):
    """Return one unambiguous file under root, serialized with portable separators."""
    root = Path(root).resolve()
    sid = meta.get("id")
    if not isinstance(sid, str) or not sid.strip():
        raise ValueError("source metadata requires a nonempty id")
    raw = meta.get("wav_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{sid}: missing wav_path")
    path = (root / raw.replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{sid}: audio path is outside processed root")
    if not path.is_file():
        candidates = [p.resolve() for p in root.rglob(f"{sid}.wav") if p.is_file()]
        if len(candidates) != 1 or not candidates[0].is_relative_to(root):
            raise ValueError(f"{sid}: expected one matching WAV, found {len(candidates)}")
        path = candidates[0]
    return path.relative_to(root).as_posix()


def build_record_fields(labels, meta, sample_id):
    """Embed canonical labels and source/channel metadata; never read at inference."""
    leaf = labels["L3"]
    if leaf not in LEAF_PARENTS:
        raise ValueError(f"{sample_id}: unsupported class {leaf!r}")
    l1, l2 = LEAF_PARENTS[leaf]
    gt = {"L1": l1, "L2": l2, "L3": leaf}
    details = deepcopy(meta)
    details.update(gt)
    details["source_id"] = re.sub(r"_ch\d+$", "", sample_id, flags=re.IGNORECASE)
    bo = meta.get("bellhop_output") or {}
    field = "tl_db" if l1 == "active" else "snr_db"
    if bo.get(field) is not None:
        value = float(bo[field])
        if not math.isfinite(value):
            raise ValueError(f"{sample_id}: non-finite {field}")
        details[field] = value
        details["quality_metric"] = ("pre_normalization_channel_energy_gain_db" if l1 == "active"
                                     else "source_pre_channel_line_spectrum_snr_db")
    ssp = (meta.get("bellhop_env") or {}).get("ssp") or {}
    depths, speeds = ssp.get("depths_m"), ssp.get("sound_speeds_mps")
    if depths is not None and speeds is not None:
        if len(depths) != len(speeds):
            raise ValueError(f"{sample_id}: unequal SSP array lengths")
        if len(depths) > 1:
            z, c = list(map(float, depths)), list(map(float, speeds))
            if not all(math.isfinite(x) for x in z + c) or any(b <= a for a, b in zip(z, z[1:])):
                raise ValueError(f"{sample_id}: invalid SSP coordinates")
            gradients = [(c[i+1]-c[i])/(z[i+1]-z[i]) for i in range(len(z)-1)]
            mean = sum(gradients) / len(gradients)
            details["ssp_complexity"] = round(sum((g-mean)**2 for g in gradients)/len(gradients), 8)
    return {"record_schema_version": RECORD_SCHEMA_VERSION, "_gt": gt, "_meta": details}
