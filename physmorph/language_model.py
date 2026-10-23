"""Project-local Qwen inference; the parser consumes only its text and metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from physmorph.text_request import ROOT, extraction_instructions

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_DIRECTORY = ROOT / ".cache" / "models" / f"{MODEL_ID.split('/')[-1]}-{MODEL_REVISION}"


class LocalQwen:
    def __init__(self, *, model_id: str = MODEL_ID, revision: str = MODEL_REVISION,
                 load_in_4bit: bool = False, enable_thinking: bool | None = None,
                 instructions: str | None = None, max_new_tokens: int = 192,
                 model_directory: str | Path | None = None):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

        if not torch.cuda.is_available():
            raise RuntimeError("The language baseline requires the local CUDA GPU.")
        model_directory = (Path(model_directory).resolve() if model_directory is not None else
                           ROOT / ".cache" / "models" / f"{model_id.split('/')[-1]}-{revision}")
        if not (model_directory / 'config.json').is_file():
            raise FileNotFoundError('Local model missing: '+str(model_directory)+
                                   '; run scripts/download_language_model.py or supply --model-dir.')
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)
        loading = {}
        if load_in_4bit:
            loading = {"device_map": {"": 0}, "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_directory, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", **loading)
        if not load_in_4bit:
            self.model = self.model.to("cuda")
        self.model = self.model.eval()
        self.template_options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        mode = "model_default"
        if enable_thinking is not None:
            mode = "thinking" if enable_thinking else "non-thinking"
        elif model_id == MODEL_ID:
            mode = "non-thinking"
        self.instructions = extraction_instructions() if instructions is None else instructions
        self.generation = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False,
                                           eos_token_id=self.tokenizer.eos_token_id,
                                           pad_token_id=self.tokenizer.pad_token_id,
                                           bos_token_id=self.model.generation_config.bos_token_id)
        self.metadata = {"model_id": model_id, "revision": revision,
                         "model_directory": str(model_directory), "device": "cuda",
                         "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16",
                         "mode": mode,
                         "chat_template_options": self.template_options,
                         "quantization": loading["quantization_config"].to_dict() if load_in_4bit else None,
                         "generation": {**self.generation.to_diff_dict(),
                                                                    "do_sample": False, "use_model_defaults": False},
                         "transformers": transformers.__version__, "torch": torch.__version__,
                         "system_prompt": self.instructions,
                         "system_prompt_sha256": hashlib.sha256(self.instructions.encode()).hexdigest()}
        if load_in_4bit:
            self.metadata.update({name: importlib.metadata.version(name) for name in ("bitsandbytes", "accelerate")})

    def __call__(self, prompt: str, *, instructions: str | None = None) -> tuple[str, dict]:
        import torch

        task = self.instructions if instructions is None else instructions
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": task}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, **self.template_options)
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation, use_model_defaults=False)
        response = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return response, {**self.metadata, 'input': prompt, 'system_prompt': task,
                          'system_prompt_sha256': hashlib.sha256(task.encode()).hexdigest(), 'response': response}
