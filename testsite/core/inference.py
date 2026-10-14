"""Unified model inference interface.

Supported backends:/n  - mock:             synthetic output, exercises the framework
  - qwen2_audio:      Qwen2-Audio-7B-Instruct
  - aero1_audio:      Aero-1-Audio (1.5B, LMMs-Lab)
  - voxtral_mini:     Voxtral-Mini-3B-2507 (Mistral)
  - voxtral_small:    Voxtral-Small-24B-2507 (Mistral)
  - af_next:          Audio-Flamingo-Next (8B, NVIDIA)
  - kimi_audio:       Kimi-Audio-7B-Instruct (Moonshot, ~10B)
  - qwen25_omni:      Qwen2.5-Omni-7B
  - minicpmo:         MiniCPM-o 2.6
  - gemma4:           Gemma 4 E2B IT (Google)
  - midashenglm:      MiDashengLM-7B-1021-BF16 (Xiaomi)
"""
import random


class ModelInference:
    """Unified inference interface; dispatches to a concrete backend via config['model']['backend']."""

    BACKENDS = {}  # populated at module bottom

    def __init__(self, config: dict):
        model_cfg = config["model"]
        self.backend = model_cfg.get("backend", "mock")
        cls = self.BACKENDS.get(self.backend)
        if cls is None:
            raise ValueError(f"Unknown backend: {self.backend}. Available: {list(self.BACKENDS)}")
        self._model = cls(model_cfg)

    def generate(self, audio_path: str, prompt: str,
                 max_new_tokens: int = None, temperature: float = None,
                 no_audio: bool = False) -> str:
        return self._model.generate(audio_path, prompt, max_new_tokens,
                                     temperature, no_audio=no_audio)

    def chat(self, audio_path: str, conversations: list,
             max_new_tokens: int = None, temperature: float = None,
             no_audio: bool = False) -> str:
        return self._model.chat(audio_path, conversations, max_new_tokens,
                                 temperature, no_audio=no_audio)

    def generate_batch(self, audio_paths: list, prompts: list,
                       max_new_tokens: int = None, temperature: float = None,
                       no_audio: bool = False) -> list:
        return self._model.generate_batch(audio_paths, prompts, max_new_tokens,
                                          temperature, no_audio=no_audio)

    def chat_batch(self, audio_paths: list, conversations: list,
                   max_new_tokens: int = None, temperature: float = None,
                   no_audio: bool = False) -> list:
        return self._model.chat_batch(audio_paths, conversations, max_new_tokens,
                                      temperature, no_audio=no_audio)


# ============================================================
# Shared inference logic (the part common to generate / chat)
# ============================================================
class _BaseAudioBackend:
    """Base class for audio LLM backends. Subclasses only implement _load_model and _load_processor."""

    _default_sampling_rate: int = 16000

    def __init__(self, model_cfg: dict):
        import torch

        self.model_id = model_cfg.get("model_id")
        if not self.model_id:
            raise ValueError(f"{self.__class__.__name__}: model_id is required")

        self.device = model_cfg.get("device", "cuda:0")
        self.torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self.default_temperature = model_cfg.get("temperature", 0.0)
        self.attn_implementation = model_cfg.get("attn_implementation")

        print(f"  Loading {self.__class__.__name__}: {self.model_id}")
        print(f"  device={self.device}, dtype={self.torch_dtype}")

        self.model = self._load_model()
        self.processor = self._load_processor()
        self.sampling_rate = getattr(
            getattr(self.processor, "feature_extractor", None), "sampling_rate",
            self._default_sampling_rate,
        )
        print(f"  Model loaded, sample_rate={self.sampling_rate}Hz")

    def _load_model(self):
        raise NotImplementedError

    def _load_processor(self):
        raise NotImplementedError

    # ---- generate / chat ----
    def generate(self, audio_path: str, prompt: str,
                 max_new_tokens: int = None, temperature: float = None,
                 no_audio: bool = False) -> str:
        max_t = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.default_temperature
        return self._infer(audio_path, prompt, None, max_t, temp, no_audio)

    def chat(self, audio_path: str, conversations: list,
             max_new_tokens: int = None, temperature: float = None,
             no_audio: bool = False) -> str:
        max_t = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.default_temperature
        return self._infer(audio_path, None, conversations, max_t, temp, no_audio)

    def generate_batch(self, audio_paths, prompts, max_new_tokens=None,
                       temperature=None, no_audio=False):
        return [self.generate(a, p, max_new_tokens, temperature, no_audio)
                for a, p in zip(audio_paths, prompts)]

    def chat_batch(self, audio_paths, conversations, max_new_tokens=None,
                   temperature=None, no_audio=False):
        return [self.chat(a, c, max_new_tokens, temperature, no_audio)
                for a, c in zip(audio_paths, conversations)]

    def _infer(self, audio_path, prompt, conversations,
               max_new_tokens, temperature, no_audio):
        """Unified inference entry point: build conversation -> processor -> model.generate -> decode."""
        import torch
        import librosa

        # 1. Build the conversation
        if conversations is not None:
            conv = []
            for i, turn in enumerate(conversations):
                role = "user" if turn["from"] == "human" else "assistant"
                if i == 0 and role == "user":
                    content = [{"type": "text", "text": turn["value"]}]
                    if not no_audio:
                        content.insert(0, {"type": "audio", "audio_url": audio_path})
                    conv.append({"role": "user", "content": content})
                elif role == "user":
                    conv.append({"role": "user", "content": [{"type": "text", "text": turn["value"]}]})
                else:
                    conv.append({"role": "assistant", "content": turn["value"]})
        else:
            content = [{"type": "text", "text": prompt}]
            if not no_audio:
                content.insert(0, {"type": "audio", "audio_url": audio_path})
            conv = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False)

        # 2. Processor
        if no_audio:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        else:
            audio_array, _ = librosa.load(audio_path, sr=self.sampling_rate)
            inputs = self._call_processor(text, audio_array)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

        # 3. Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # 4. Decode
        return self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def _call_processor(self, text: str, audio_array):
        """Call the processor with text + audio. Subclasses may override this for different APIs.
        Default: audios=[audio_array] (Qwen2-Audio style)"""
        return self.processor(
            text=text, audios=[audio_array],
            return_tensors="pt", sampling_rate=self.sampling_rate,
        )


# ============================================================
# Forward-kwarg compatibility shim
# ============================================================
def _filter_unsupported_forward_kwargs(model, label):
    """Drop generation kwargs the model ``forward`` does not declare.

    transformers 4.54+ passes ``cache_position`` into ``model(**inputs)`` during
    ``generate()``. Checkpoints whose ``forward`` predates that (Qwen2-Audio,
    Aero-1-Audio) raise ``TypeError``. The wrapped language model recomputes
    positions from the KV cache when the argument is absent, so silently
    dropping undeclared kwargs does not change decoding.
    """
    import functools as _functools
    import inspect as _inspect

    original = model.forward
    allowed = set(_inspect.signature(original).parameters)
    warned = {"done": False}

    @_functools.wraps(original)
    def forward(*args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        dropped = set(kwargs) - set(filtered)
        if dropped and not warned["done"]:
            warned["done"] = True
            print(f"  [{label}] compat shim dropped unsupported forward kwargs: "
                  f"{sorted(dropped)}")
        return original(*args, **filtered)

    # functools.wraps keeps __wrapped__, so inspect.signature() (used by
    # transformers' _validate_model_kwargs) still sees the real signature.
    model.forward = forward
    return model


# ============================================================
# Qwen2-Audio-7B
# ============================================================
class Qwen2AudioBackend(_BaseAudioBackend):
    """Qwen2-Audio-7B-Instruct (Alibaba)"""

    def _load_model(self):
        from transformers import Qwen2AudioForConditionalGeneration
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.startswith("cuda") else self.device,
            low_cpu_mem_usage=True,
        )
        return _filter_unsupported_forward_kwargs(model, "qwen2_audio")

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id)


# ============================================================
# Aero-1-Audio (1.5B, LMMs-Lab)
# https://huggingface.co/lmms-lab/Aero-1-Audio
# ============================================================
class Aero1AudioBackend(_BaseAudioBackend):
    """Aero-1-Audio by LMMs-Lab. Qwen2.5-1.5B backbone, 50k hrs curated audio."""

    def _load_model(self):
        from transformers import AutoModelForCausalLM
        # The checkpoint's remote modelling_aero.py imports
        # ``Qwen2AudioFlashAttention2``, which transformers 4.54 removed. It is
        # only referenced when _attn_implementation == "flash_attention_2"
        # (this run uses sdpa), so expose a placeholder subclass — never an
        # alias of the real attention class, to avoid clobbering its forward.
        import transformers.models.qwen2_audio.modeling_qwen2_audio as _qa
        if not hasattr(_qa, "Qwen2AudioFlashAttention2"):
            class _Qwen2AudioFlashAttention2Shim(_qa.Qwen2AudioAttention):
                """Placeholder for a class removed in transformers 4.54."""

            _qa.Qwen2AudioFlashAttention2 = _Qwen2AudioFlashAttention2Shim
            print("  [aero1_audio] injected Qwen2AudioFlashAttention2 shim")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.startswith("cuda") else self.device,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        return _filter_unsupported_forward_kwargs(model, "aero1_audio")

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)


# ============================================================
# Voxtral Mini 3B / Small 24B (Mistral AI)
# https://huggingface.co/docs/transformers/model_doc/voxtral
# ============================================================
class VoxtralBackend(_BaseAudioBackend):
    """Official Voxtral backend with native conversation batching."""

    def _load_model(self):
        from transformers import VoxtralForConditionalGeneration
        kwargs = dict(
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.startswith("cuda") else self.device,
            low_cpu_mem_usage=True,
        )
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return VoxtralForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs).eval()

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id)

    @staticmethod
    def _conversation(audio_path, prompt=None, history=None, no_audio=False):
        """Convert evaluation history to the official Voxtral chat schema."""
        conv = []
        source = history or [{"from": "human", "value": prompt}]
        for i, turn in enumerate(source):
            role = "user" if turn["from"] == "human" else "assistant"
            if role == "assistant":
                content = turn["value"]
            else:
                content = [{"type": "text", "text": turn["value"]}]
                if i == 0 and not no_audio:
                    content.insert(0, {"type": "audio", "path": audio_path})
            conv.append({"role": role, "content": content})
        return conv

    def _batch(self, audio_paths, prompts=None, histories=None,
               max_new_tokens=None, temperature=None, no_audio=False):
        """Run a real tensor batch through apply_chat_template and generate."""
        import torch

        conversations = [self._conversation(
            audio_path,
            prompt=None if prompts is None else prompts[i],
            history=None if histories is None else histories[i],
            no_audio=no_audio,
        ) for i, audio_path in enumerate(audio_paths)]
        inputs = self.processor.apply_chat_template(conversations)
        inputs = inputs.to(self.device, dtype=self.torch_dtype)
        temp = self.default_temperature if temperature is None else temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temp if temp > 0 else None,
                do_sample=temp > 0,
            )

        return [text.strip() for text in self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )]

    def generate_batch(self, audio_paths, prompts, max_new_tokens=None,
                       temperature=None, no_audio=False):
        return self._batch(
            audio_paths, prompts=prompts, max_new_tokens=max_new_tokens,
            temperature=temperature, no_audio=no_audio)

    def chat_batch(self, audio_paths, conversations, max_new_tokens=None,
                   temperature=None, no_audio=False):
        return self._batch(
            audio_paths, histories=conversations, max_new_tokens=max_new_tokens,
            temperature=temperature, no_audio=no_audio)

    def _infer(self, audio_path, prompt, conversations,
               max_new_tokens, temperature, no_audio):
        return self._batch(
            [audio_path],
            None if conversations is not None else [prompt],
            [conversations] if conversations is not None else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            no_audio=no_audio,
        )[0]


# Backward-compatible import name used by older local scripts.
VoxtralMiniBackend = VoxtralBackend


# ============================================================
# Qwen2.5-Omni-7B
# https://huggingface.co/Qwen/Qwen2.5-Omni-7B
# ============================================================
class Qwen25OmniBackend(_BaseAudioBackend):
    """Official Qwen2.5-Omni Transformers backend with native batching."""

    def _load_model(self):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        kwargs = dict(torch_dtype=self.torch_dtype, device_map=self.device,
                      low_cpu_mem_usage=True)
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs).eval()

    def _load_processor(self):
        from transformers import Qwen2_5OmniProcessor
        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        # Batched decoder-only generation must left-pad prompts of different
        # lengths so generation always starts after the final real token.
        processor.tokenizer.padding_side = "left"
        # The checkpoint ships a generation_config.json that only carries
        # "_from_model_config": its eos/pad ids live in the nested
        # thinker_config, so GenerationConfig ends up with eos_token_id=None.
        # generate() then has no stop token and decodes the whole
        # max_new_tokens budget (hallucinated follow-up "Human:" turns), which
        # the single-letter answer parser rejects -> every metric becomes 0.
        # Fill the ids from the tokenizer so decoding stops at <|im_end|>.
        tokenizer = processor.tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        self.model.generation_config.eos_token_id = eos_id
        self.model.generation_config.pad_token_id = pad_id
        print(f"  generation stop ids: eos={eos_id} pad={pad_id}")
        return processor

    @staticmethod
    def _conversation(audio_path, prompt=None, history=None, no_audio=False):
        conv = []
        source = history or [{"from": "human", "value": prompt}]
        for i, turn in enumerate(source):
            role = "user" if turn["from"] == "human" else "assistant"
            content = [{"type": "text", "text": turn["value"]}]
            if i == 0 and role == "user" and not no_audio:
                content.insert(0, {"type": "audio", "audio": audio_path})
            conv.append({"role": role, "content": content})
        return conv

    def _batch(self, audio_paths, prompts=None, histories=None,
               max_new_tokens=None, temperature=None, no_audio=False):
        import torch
        from qwen_omni_utils import process_mm_info
        conversations = [self._conversation(a,
                         prompt=None if prompts is None else prompts[i],
                         history=None if histories is None else histories[i],
                         no_audio=no_audio) for i, a in enumerate(audio_paths)]
        texts = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
        inputs = self.processor(text=texts, audio=audios, images=images, videos=videos,
                                padding=True, return_tensors="pt",
                                use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        temp = self.default_temperature if temperature is None else temperature
        # Belt and braces: pass the stop ids explicitly as well, so decoding is
        # bounded even if a future transformers version ignores the patched
        # generation_config or the checkpoint's empty one is reloaded.
        tokenizer = self.processor.tokenizer
        eos_id = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_id is None:
            eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        with torch.inference_mode():
            ids = self.model.generate(
                **inputs, use_audio_in_video=False,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        completion = ids[:, inputs["input_ids"].shape[1]:]
        return [x.strip() for x in self.processor.batch_decode(
            completion, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)]

    def generate_batch(self, audio_paths, prompts, max_new_tokens=None,
                       temperature=None, no_audio=False):
        return self._batch(audio_paths, prompts=prompts, max_new_tokens=max_new_tokens,
                           temperature=temperature, no_audio=no_audio)

    def chat_batch(self, audio_paths, conversations, max_new_tokens=None,
                   temperature=None, no_audio=False):
        return self._batch(audio_paths, histories=conversations,
                           max_new_tokens=max_new_tokens, temperature=temperature,
                           no_audio=no_audio)

    def _infer(self, audio_path, prompt, conversations, max_new_tokens,
               temperature, no_audio):
        return self._batch([audio_path], None if conversations else [prompt],
                           [conversations] if conversations else None,
                           max_new_tokens, temperature, no_audio)[0]


# ============================================================
# MiniCPM-o 2.6
# https://github.com/IsiSinclair/MiniCPM-o-2.6
# ============================================================
class MiniCPMoBackend(_BaseAudioBackend):
    """Official MiniCPM-o ``model.chat`` audio-to-text backend.

    The released chat API is single-example; inherited batch methods deliberately
    loop instead of pretending to provide tensor batching.
    """

    def _load_model(self):
        import transformers
        if transformers.__version__ != "4.44.2":
            raise RuntimeError(
                "MiniCPM-o 2.6 requires transformers==4.44.2 (official pin); "
                "install the official MiniCPM-o dependencies before using this backend"
            )
        from transformers import AutoModel
        kwargs = dict(trust_remote_code=True, torch_dtype=self.torch_dtype,
                      low_cpu_mem_usage=True)
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        model = AutoModel.from_pretrained(self.model_id, **kwargs).eval()
        if self.device == "auto":
            # Remote-code chat does not reliably support Accelerate device maps.
            self.device = "cuda:0"
        return model.to(self.device)

    def _load_processor(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    def _infer(self, audio_path, prompt, conversations, max_new_tokens,
               temperature, no_audio):
        import librosa
        history = conversations or [{"from": "human", "value": prompt}]
        msgs = []
        for i, turn in enumerate(history):
            role = "user" if turn["from"] == "human" else "assistant"
            content = [turn["value"]]
            if i == 0 and role == "user" and not no_audio:
                audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                content.append(audio)
            msgs.append({"role": role, "content": content})
        temp = self.default_temperature if temperature is None else temperature
        answer = self.model.chat(
            msgs=msgs, tokenizer=self.processor,
            sampling=temp > 0, temperature=temp if temp > 0 else 0.1,
            max_new_tokens=max_new_tokens,
            use_tts_template=False, generate_audio=False,
        )
        if isinstance(answer, dict):
            answer = answer.get("text", "")
        return str(answer or "").strip()


# ============================================================
# Audio-Flamingo-Next (8B, NVIDIA)
# https://huggingface.co/nvidia/audio-flamingo-next-hf
# ============================================================
class AudioFlamingoNextBackend(_BaseAudioBackend):
    """Current official AF-Next HF backend with conversation batching."""

    def _load_model(self):
        from transformers import AutoModel

        kwargs = dict(dtype=self.torch_dtype, device_map=self.device,
                      low_cpu_mem_usage=True)
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return AutoModel.from_pretrained(
            self.model_id, **kwargs
        ).eval()

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id)

    @staticmethod
    def _conversation(audio_path, prompt=None, history=None, no_audio=False):
        conv = []
        source = history or [{"from": "human", "value": prompt}]
        for i, turn in enumerate(source):
            role = "user" if turn["from"] == "human" else "assistant"
            content = [{"type": "text", "text": turn["value"]}]
            if i == 0 and role == "user" and not no_audio:
                content.append({"type": "audio", "path": audio_path})
            conv.append({"role": role, "content": content})
        return conv

    def _batch(self, audio_paths, prompts=None, histories=None,
               max_new_tokens=None, temperature=None, no_audio=False):
        import torch
        conversations = [self._conversation(
            a, None if prompts is None else prompts[i],
            None if histories is None else histories[i], no_audio)
            for i, a in enumerate(audio_paths)]
        batch = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device, dtype=self.model.dtype)
        temp = self.default_temperature if temperature is None else temperature
        with torch.inference_mode():
            generated = self.model.generate(
                **batch, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=temp > 0, temperature=temp if temp > 0 else None,
                repetition_penalty=1.2)
        completion = generated[:, batch["input_ids"].shape[1]:]
        return [x.strip() for x in self.processor.batch_decode(
            completion, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)]

    def generate_batch(self, audio_paths, prompts, max_new_tokens=None,
                       temperature=None, no_audio=False):
        return self._batch(audio_paths, prompts=prompts, max_new_tokens=max_new_tokens,
                           temperature=temperature, no_audio=no_audio)

    def chat_batch(self, audio_paths, conversations, max_new_tokens=None,
                   temperature=None, no_audio=False):
        return self._batch(audio_paths, histories=conversations,
                           max_new_tokens=max_new_tokens, temperature=temperature,
                           no_audio=no_audio)

    def _infer(self, audio_path, prompt, conversations, max_new_tokens,
               temperature, no_audio):
        return self._batch([audio_path], None if conversations else [prompt],
                           [conversations] if conversations else None,
                           max_new_tokens, temperature, no_audio)[0]


# ============================================================
# Gemma 4 E2B IT (Google)
# https://huggingface.co/google/gemma-4-E2B-it
# ============================================================
class Gemma4Backend(_BaseAudioBackend):
    """Official Gemma 4 audio backend with native conversation batching."""

    def _load_model(self):
        from transformers import AutoModelForMultimodalLM

        kwargs = dict(dtype=self.torch_dtype, device_map=self.device,
                      low_cpu_mem_usage=True)
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        return AutoModelForMultimodalLM.from_pretrained(
            self.model_id, **kwargs
        ).eval()

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(
            self.model_id, padding_side="left"
        )

    @staticmethod
    def _conversation(audio_path, prompt=None, history=None, no_audio=False):
        conv = []
        source = history or [{"from": "human", "value": prompt}]
        for i, turn in enumerate(source):
            role = "user" if turn["from"] == "human" else "assistant"
            if role == "assistant":
                content = turn["value"]
            else:
                content = [{"type": "text", "text": turn["value"]}]
                # Gemma 4 recommends placing audio after the accompanying text.
                if i == 0 and not no_audio:
                    content.append({"type": "audio", "audio": audio_path})
            conv.append({"role": role, "content": content})
        return conv

    def _batch(self, audio_paths, prompts=None, histories=None,
               max_new_tokens=None, temperature=None, no_audio=False):
        import torch

        conversations = [self._conversation(
            audio_path,
            None if prompts is None else prompts[i],
            None if histories is None else histories[i],
            no_audio,
        ) for i, audio_path in enumerate(audio_paths)]
        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        temp = self.default_temperature if temperature is None else temperature
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temp > 0,
        }
        if temp > 0:
            generation_kwargs["temperature"] = temp

        with torch.inference_mode():
            generated = self.model.generate(**batch, **generation_kwargs)
        completion = generated[:, batch["input_ids"].shape[1]:]
        return [text.strip() for text in self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )]

    def generate_batch(self, audio_paths, prompts, max_new_tokens=None,
                       temperature=None, no_audio=False):
        return self._batch(
            audio_paths, prompts=prompts, max_new_tokens=max_new_tokens,
            temperature=temperature, no_audio=no_audio,
        )

    def chat_batch(self, audio_paths, conversations, max_new_tokens=None,
                   temperature=None, no_audio=False):
        return self._batch(
            audio_paths, histories=conversations,
            max_new_tokens=max_new_tokens, temperature=temperature,
            no_audio=no_audio,
        )

    def _infer(self, audio_path, prompt, conversations, max_new_tokens,
               temperature, no_audio):
        return self._batch(
            [audio_path],
            None if conversations else [prompt],
            [conversations] if conversations else None,
            max_new_tokens,
            temperature,
            no_audio,
        )[0]


# ============================================================
# MiDashengLM-7B-1021-BF16 (Xiaomi)
# https://huggingface.co/mispeech/midashenglm-7b-1021-bf16
# ============================================================
class MiDashengLMBackend(_BaseAudioBackend):
    """MiDashengLM backend with native batched audio conversations."""

    # The checkpoint expands one <|AUDIO|> placeholder per projected Dasheng
    # frame. With the model's 160-sample hop, 4x encoder subsampling and 5x
    # projector subsampling, inputs shorter than 3040 samples produce zero
    # placeholders. Use a round 0.20 s floor at 16 kHz so short pulse clips
    # still produce one audio span. Padding is applied in memory only.
    _MIN_AUDIO_SECONDS = 0.20

    def _load_model(self):
        import torch
        import torchaudio.functional as audio_functional
        from transformers import AutoModelForCausalLM

        kwargs = dict(
            dtype=self.torch_dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        # Transformers 5 initializes custom models on the meta device. The
        # Dasheng frontend creates two non-persistent signal-processing buffers
        # during __init__, while torchaudio creates part of its mel filter bank
        # explicitly on CPU. That CPU/meta mixture raises before weights load.
        # Force just these two buffers onto CPU during construction, then move
        # them with the fully loaded model to the requested CUDA device.
        original_hann_window = torch.hann_window
        original_melscale_fbanks = audio_functional.melscale_fbanks

        def cpu_hann_window(*args, **kwargs):
            kwargs["device"] = "cpu"
            with torch.device("cpu"):
                return original_hann_window(*args, **kwargs)

        def cpu_melscale_fbanks(*args, **kwargs):
            with torch.device("cpu"):
                return original_melscale_fbanks(*args, **kwargs)

        torch.hann_window = cpu_hann_window
        audio_functional.melscale_fbanks = cpu_melscale_fbanks
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **kwargs
            )
        finally:
            torch.hann_window = original_hann_window
            audio_functional.melscale_fbanks = original_melscale_fbanks

        return model.to(self.device).eval()

    def _load_processor(self):
        import transformers
        from transformers import AutoProcessor

        # The checkpoint's processor was authored for Transformers 4.x and
        # imports these classes from the package root. Transformers 5 keeps the
        # implementations but no longer exports the legacy names there.
        if not hasattr(transformers, "Qwen2Tokenizer"):
            from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
            transformers.Qwen2Tokenizer = Qwen2Tokenizer
        if not hasattr(transformers, "Qwen2TokenizerFast"):
            transformers.Qwen2TokenizerFast = transformers.Qwen2Tokenizer
        if not hasattr(transformers, "Wav2Vec2FeatureExtractor"):
            from transformers.models.wav2vec2.feature_extraction_wav2vec2 import (
                Wav2Vec2FeatureExtractor,
            )
            transformers.Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor

        return AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

    def _prepare_audio(self, audio_path):
        import librosa
        import numpy as np

        sampling_rate = int(self.processor.sampling_rate)
        audio, _ = librosa.load(
            audio_path,
            sr=sampling_rate,
            mono=True,
            dtype=np.float32,
        )
        minimum_samples = int(round(self._MIN_AUDIO_SECONDS * sampling_rate))
        if audio.shape[0] < minimum_samples:
            audio = np.pad(audio, (0, minimum_samples - audio.shape[0]))
        return audio

    @staticmethod
    def _conversation(audio, prompt=None, history=None, no_audio=False):
        conv = []
        source = history or [{"from": "human", "value": prompt}]
        for i, turn in enumerate(source):
            role = "user" if turn["from"] == "human" else "assistant"
            if role == "assistant":
                content = turn["value"]
            else:
                content = [{"type": "text", "text": turn["value"]}]
                if i == 0 and not no_audio:
                    # Match the official template: accompanying text, then audio.
                    content.append({"type": "audio", "audio": audio})
            conv.append({"role": role, "content": content})
        return conv

    def _batch(self, audio_paths, prompts=None, histories=None,
               max_new_tokens=None, temperature=None, no_audio=False):
        import torch

        audios = (audio_paths if no_audio else
                  [self._prepare_audio(path) for path in audio_paths])
        conversations = [self._conversation(
            audio,
            None if prompts is None else prompts[i],
            None if histories is None else histories[i],
            no_audio,
        ) for i, audio in enumerate(audios)]

        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        temp = self.default_temperature if temperature is None else temperature
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": temp > 0,
        }
        if temp > 0:
            generation_kwargs["temperature"] = temp

        with torch.inference_mode():
            generated = self.model.generate(**batch, **generation_kwargs)

        # MiDashengLM internally calls decoder.generate(inputs_embeds=...).
        # Its return contains generated token IDs only, unlike the usual
        # decoder-only output that prepends input_ids. Do not slice by prompt
        # length here or the beginning of every answer will be discarded.
        return [text.strip() for text in self.processor.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )]

    def generate_batch(self, audio_paths, prompts, max_new_tokens=None,
                       temperature=None, no_audio=False):
        return self._batch(
            audio_paths,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            no_audio=no_audio,
        )

    def chat_batch(self, audio_paths, conversations, max_new_tokens=None,
                   temperature=None, no_audio=False):
        return self._batch(
            audio_paths,
            histories=conversations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            no_audio=no_audio,
        )

    def _infer(self, audio_path, prompt, conversations, max_new_tokens,
               temperature, no_audio):
        return self._batch(
            [audio_path],
            None if conversations else [prompt],
            [conversations] if conversations else None,
            max_new_tokens,
            temperature,
            no_audio,
        )[0]


# ============================================================
# Kimi-Audio-7B-Instruct (Moonshot AI, ~10B)
# https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
# ============================================================
class KimiAudioBackend(_BaseAudioBackend):
    """Kimi-Audio-7B-Instruct by Moonshot AI. 13M hrs pretraining.

    Uses the official kimia_infer API instead of the transformers processor path.
    Requires: pip install kimia-infer (or install it from GitHub)
    """

    def _load_model(self):
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError:
            raise ImportError(
                "Kimi-Audio requires the kimia-infer package.\n"
                "  pip install git+https://github.com/MoonshotAI/Kimi-Audio.git\n"
                "or download it from ModelScope and install locally: pip install -e /path/to/Kimi-Audio"
            )
        self._kimia_params = {
            "audio_temperature": 0.0,
            "text_temperature": self.default_temperature,
            "output_type": "text",
        }
        return KimiAudio(model_path=self.model_id, load_detokenizer=False)

    def _load_processor(self):
        return None  # Kimi-Audio needs no processor; the model ships everything

    def _infer(self, audio_path, prompt, conversations,
               max_new_tokens, temperature, no_audio):
        """Kimi-Audio does not follow the standard generate/processor flow; the whole inference path is overridden."""
        # Build messages in Kimi-Audio format
        if conversations is not None:
            messages = []
            for turn in conversations:
                role = turn["from"] == "human" and "user" or "assistant"
                if role == "user":
                    messages.append({
                        "role": "user", "message_type": "text",
                        "content": turn["value"],
                    })
                    if turn == conversations[0] and not no_audio:
                        messages.append({
                            "role": "user", "message_type": "audio",
                            "content": audio_path,
                        })
                else:
                    messages.append({
                        "role": "assistant", "message_type": "text",
                        "content": turn["value"],
                    })
        else:
            messages = [{"role": "user", "message_type": "text", "content": prompt}]
            if not no_audio:
                messages.append({"role": "user", "message_type": "audio", "content": audio_path})

        params = dict(self._kimia_params)
        if temperature is not None:
            params["text_temperature"] = temperature
        params["output_type"] = "text"

        _, text_output = self.model.generate(messages, **params)
        return (text_output or "").strip()


# ============================================================
# MockModel (for tests)
# ============================================================
class MockModel:
    """Synthetic model used to exercise the evaluation framework."""

    L1_MAP = {"active": "an actively transmitted signal", "passive": "a source-radiated signal"}
    L2_MAP = {"pulse": "Detection pulse", "communication": "Communication signal", "ship_noise": "Ship-radiated noise"}
    L3_MAP = {
        "CW": "CW (Continuous Wave)", "LFM": "LFM (Linear Frequency Modulation)",
        "HFM": "HFM (Hyperbolic Frequency Modulation)",
        "2FSK": "2FSK (Binary Frequency Shift Keying)",
        "4FSK": "4FSK (Quaternary Frequency Shift Keying)",
        "BPSK": "BPSK (Binary Phase Shift Keying)",
        "QPSK": "QPSK (Quadrature Phase Shift Keying)",
        "OFDM": "OFDM (Orthogonal Frequency Division Multiplexing)",
        "cargo": "Cargo vessel", "cruise": "Cruise ship", "fishing": "Fishing vessel",
        "warship": "Naval vessel", "underwater_target": "Underwater vehicle",
    }
    L2_POOLS = {"active": ["pulse", "communication"], "passive": ["ship_noise"]}
    L3_BY_L2 = {
        "pulse": ["CW", "LFM", "HFM"],
        "communication": ["2FSK", "4FSK", "BPSK", "QPSK", "OFDM"],
        "ship_noise": ["cargo", "cruise", "fishing", "warship", "underwater_target"],
    }

    def __init__(self, mock_config: dict):
        self.accuracy = mock_config.get("accuracy", 0.85)
        self.l1_acc = mock_config.get("l1_accuracy", self.accuracy)
        self.l2_acc = mock_config.get("l2_accuracy", self.accuracy)
        self.l3_acc = mock_config.get("l3_accuracy", self.accuracy)
        self._rng = random.Random(42)

    def generate(self, audio_path: str, prompt: str,
                 max_new_tokens: int = 512, temperature: float = 0.1,
                 no_audio: bool = False) -> str:
        if no_audio:
            l1 = self._rng.choice(["active", "passive"])
            l2 = self._rng.choice(self.L2_POOLS[l1])
            l3 = self._random_l3(l2)
        else:
            l1, l2, l3 = self._pick_answer(audio_path)
            if self._rng.random() >= self.l1_acc:
                l1 = "passive" if l1 == "active" else "active"
                pool = self.L2_POOLS.get(l1, ["pulse"])
                l2 = self._rng.choice(pool)
                l3 = self._random_l3(l2)
            if self._rng.random() >= self.l2_acc:
                l2 = self._pick_wrong_l2(l1, l2)
                l3 = self._random_l3(l2)
            if self._rng.random() >= self.l3_acc:
                l3 = self._pick_wrong_l3(l2, l3)
        option_map = self._extract_options(prompt)
        target_letter = self._find_matching_option(option_map, l1, l2, l3)
        if target_letter:
            return f"{target_letter}"
        return self._natural_language_output(l1, l2, l3, prompt)

    def chat(self, audio_path: str, conversations: list,
             max_new_tokens: int = 512, temperature: float = 0.1,
             no_audio: bool = False) -> str:
        last_msg = ""
        for msg in reversed(conversations):
            if msg.get("from") == "human":
                last_msg = msg.get("value", "")
                break
        return self.generate(audio_path, last_msg, max_new_tokens,
                             temperature, no_audio=no_audio)

    def generate_batch(self, audio_paths, prompts, max_new_tokens=512,
                       temperature=0.1, no_audio=False):
        return [self.generate(a, p, max_new_tokens, temperature, no_audio)
                for a, p in zip(audio_paths, prompts)]

    def chat_batch(self, audio_paths, conversations, max_new_tokens=512,
                   temperature=0.1, no_audio=False):
        return [self.chat(a, c, max_new_tokens, temperature, no_audio)
                for a, c in zip(audio_paths, conversations)]

    def _extract_options(self, prompt: str) -> dict:
        import re
        option_map = {}
        for m in re.finditer(
            r'(?:^|\n)\s*([A-Za-z]+)\s*[\.\)\、\s]\s*(.+?)(?:\n|$)',
            prompt, re.MULTILINE,
        ):
            letter = m.group(1).strip()
            text = m.group(2).strip().rstrip('，,。.')
            if letter and text and len(letter) <= 2:
                option_map[letter.upper()] = text
        return option_map

    def _find_matching_option(self, option_map, l1, l2, l3):
        l1_name = self.L1_MAP.get(l1, l1)
        l2_name = self.L2_MAP.get(l2, l2)
        l3_name = self.L3_MAP.get(l3, l3)
        l1_c, l2_c, l3_c = set(), set(), set()
        for letter, text in option_map.items():
            if l1_name in text or l1 in text.lower():
                l1_c.add(letter)
            if l2_name in text or l2 in text.lower():
                l2_c.add(letter)
            if l3_name in text or l3.lower() in text.lower():
                l3_c.add(letter)
        if l3_c:
            for letter in l3_c:
                if letter in l2_c and letter in l1_c:
                    return letter
            return next(iter(l3_c))
        if l2_c:
            return next(iter(l2_c))
        if l1_c:
            return next(iter(l1_c))
        return ""

    def _natural_language_output(self, l1, l2, l3, prompt):
        if self._is_l1_question(prompt):
            return f"This is {self.L1_MAP[l1]}."
        l2_name = self.L2_MAP[l2]
        l3_name = self.L3_MAP.get(l3, l3)
        return f"This is {self.L1_MAP[l1]}, specifically a {l3_name} ({l2_name}). Based on time-frequency analysis, the signal exhibits characteristic {l3_name} acoustic features."

    @staticmethod
    def _is_l1_question(prompt):
        has_active = any(w in prompt.lower() for w in ["active", "transmitted", "transmission"])
        has_passive = any(w in prompt.lower() for w in ["passive", "received", "listening"])
        has_detail = any(w in prompt.lower() for w in ["specific", "subcategory", "further", "which type", "exact"])
        return has_active and has_passive and not has_detail

    def _pick_answer(self, audio_path):
        import os
        fname = os.path.basename(audio_path).lower()
        path_lower = audio_path.lower()
        if "05_ship" in path_lower:
            for l3 in self.L3_BY_L2["ship_noise"]:
                if l3 in path_lower:
                    return ("passive", "ship_noise", l3)
        if "pulsecom" in path_lower:
            for l2 in ("pulse", "communication"):
                for l3 in self.L3_BY_L2[l2]:
                    if l3.lower() in path_lower:
                        return ("active", l2, l3)
        for l3 in self.L3_BY_L2["ship_noise"]:
            if l3 in fname:
                return ("passive", "ship_noise", l3)
        for l2 in ("pulse", "communication"):
            for l3 in self.L3_BY_L2[l2]:
                if l3.lower() in fname:
                    return ("active", l2, l3)
        return ("active", "pulse", "CW")

    def _random_l3(self, l2):
        return self._rng.choice(self.L3_BY_L2.get(l2, ["CW"]))

    def _pick_wrong_l2(self, l1, correct_l2):
        pool = [s for s in self.L2_POOLS.get(l1, []) if s != correct_l2]
        return self._rng.choice(pool) if pool else correct_l2

    def _pick_wrong_l3(self, l2, correct_l3):
        pool = [s for s in self.L3_BY_L2.get(l2, []) if s != correct_l3]
        return self._rng.choice(pool) if pool else correct_l3


# ============================================================
# Backend registry -> ModelInference dispatch table
# ============================================================
ModelInference.BACKENDS = {
    "mock":             MockModel,
    "qwen2_audio":      Qwen2AudioBackend,
    "aero1_audio":      Aero1AudioBackend,
    "voxtral_mini":     VoxtralBackend,
    "voxtral_small":    VoxtralBackend,
    "af_next":          AudioFlamingoNextBackend,
    "kimi_audio":       KimiAudioBackend,
    "qwen25_omni":      Qwen25OmniBackend,
    "minicpmo":         MiniCPMoBackend,
    "gemma4":           Gemma4Backend,
    "midashenglm":      MiDashengLMBackend,
}
