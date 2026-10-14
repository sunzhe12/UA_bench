"""
Load self-contained Step 2 JSONL records and processed WAV files.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .dataset_record import validate_gt, RECORD_SCHEMA_VERSION
from .integrity import read_manifest


@dataclass
class EvalSample:
    """One evaluation sample."""
    sample_id: str
    audio_path: str                       # path relative to audio_root
    gt: Dict[str, str]                    # {"L1":"active","L2":"pulse","L3":"LFM"}
    metadata: Dict[str, Any] = field(default_factory=dict)
    questions: tuple = ()
    qa_prompt_version: str = "dataset_unversioned"

    def validate_questions(self):
        if len(self.questions) != 3 or any(
            not isinstance(q, str) or not q.strip() for q in self.questions
        ):
            raise ValueError(f"{self.sample_id}: dataset must provide three nonempty human questions")


class DataLoader:
    """
    Load labels, questions and diagnostic metadata from fields embedded in the Step 2 JSONL.

    JSONL format (Step 2 output):
      {"id":"lfm_027998_ch0","audio":"PulseCom/wav_by_type/LFM/...","conversations":[...]}

    GT source: embedded _gt only. No original metadata files are required.
    """

    def __init__(self, config: dict):
        pass

    def load(
        self,
        jsonl_path: str,
        audio_root: str,
        limit: int = None,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> List[EvalSample]:
        """Load evaluation samples."""
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("invalid shard index/count")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        samples = []
        audio_base = Path(audio_root)
        records = read_manifest(jsonl_path)
        selected = [r for i, r in enumerate(records) if i % num_shards == shard_index]
        if limit is not None:
            selected = selected[:limit]
        for item in selected:
            sample = self._build_sample(item, audio_base)
            if sample is None:
                raise ValueError(f"{jsonl_path}: {item['id']}: sample could not be loaded")
            if not Path(sample.audio_path).is_file():
                raise FileNotFoundError(f"{sample.sample_id}: missing audio file: {sample.audio_path}")
            samples.append(sample)
        print(f"[Integrity] manifest={len(records)}, shard={shard_index}/{num_shards}, loaded={len(samples)}")
        return samples

    def _build_sample(self, item: dict, audio_base: Path) -> Optional[EvalSample]:
        sample_id = item.get("id", "")
        audio_rel = item.get("audio", "")

        if not sample_id or not audio_rel:
            return None

        conversations = item.get("conversations")
        if not isinstance(conversations, list) or any(not isinstance(t, dict) for t in conversations):
            raise ValueError(f"{sample_id}: missing or malformed dataset conversations")
        questions = tuple(t.get("value") for t in conversations if t.get("from") == "human")
        if len(questions) != 3 or any(not isinstance(q, str) or not q.strip() for q in questions):
            raise ValueError(f"{sample_id}: dataset must provide three nonempty human questions")

        version = item.get("record_schema_version")
        if version is not None and version != RECORD_SCHEMA_VERSION:
            raise ValueError(f"{sample_id}: unsupported record schema: {version}")
        gt = validate_gt(item.get("_gt"), sample_id)
        meta = item.get("_meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"{sample_id}: _meta must be an object")

        return EvalSample(
            sample_id=sample_id,
            audio_path=str(audio_base / audio_rel.replace("\\", "/")),
            gt=gt,
            metadata=dict(meta),
            questions=questions,
            qa_prompt_version=item.get("qa_prompt_version", "dataset_unversioned"),
        )

