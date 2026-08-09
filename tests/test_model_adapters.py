import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from llm_controllability.activations.probes import collect_residual_states_many
from llm_controllability.controllability.types import (
    ControlChannel,
    InterventionMetadata,
    StateSample,
)
from llm_controllability.features.gemma_scope import (
    gemma_scope_release,
    gemma_scope_sae_id,
    run_gemma_scope_study,
)
from llm_controllability.models import loading
from llm_controllability.models.adapters import (
    ModelProfile,
    attach_model_profile,
    last_nonpadding_indices,
    model_profile,
    prompt_control_parts,
    resolve_model_profile,
    tokenize_prompts,
)
from llm_controllability.models.architecture import get_layers
from llm_controllability.models.loading import load_model
from llm_controllability.models.runtime import model_runtime_config
from llm_controllability.reachability.io import save_state_samples


class DummyBatch(dict):
    def to(self, device):
        return DummyBatch(
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in self.items()
            }
        )


class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 0

    def __len__(self):
        return 128

    def add_special_tokens(self, values):
        self.pad_token = values["pad_token"]
        return 1

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=None,
    ):
        assert not tokenize
        suffix = "<assistant>"
        if enable_thinking is False:
            suffix += "<no_think>"
        return f"<user>{messages[0]['content']}</user>{suffix if add_generation_prompt else ''}"

    def encode(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        values = [ord(character) % 97 + 1 for character in text]
        if add_special_tokens:
            values = [99, *values]
        if truncation and max_length is not None:
            values = values[-max_length:]
        if return_tensors == "pt":
            return torch.tensor([values], dtype=torch.long)
        return values

    def __call__(self, texts, **kwargs):
        values = [texts] if isinstance(texts, str) else list(texts)
        encoded = [
            self.encode(
                text,
                add_special_tokens=kwargs.get("add_special_tokens", True),
                truncation=kwargs.get("truncation", False),
                max_length=kwargs.get("max_length"),
            )
            for text in values
        ]
        width = max(map(len, encoded))
        padded = [row + [self.pad_token_id] * (width - len(row)) for row in encoded]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in encoded]
        return DummyBatch(
            {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(mask),
            }
        )


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size):
        return size


class FakeSAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode(self, states):
        return torch.cat([torch.relu(states), torch.relu(-states)], dim=1)

    def decode(self, features):
        width = features.shape[1] // 2
        return features[:, :width] - features[:, width:]


def _sample(example_id, channel, name, state):
    return StateSample(
        example_id=example_id,
        model_name="google/gemma-3-4b-it",
        layer=1,
        intervention=InterventionMetadata(name, channel, 0.0),
        state=np.asarray(state, dtype=np.float32),
        prompt="question",
        output="answer",
        behavior_preserved=True,
    )


class ModelAdapterTests(unittest.TestCase):
    def test_recommended_profiles(self):
        gemma3 = resolve_model_profile("google/gemma-3-4b-it")
        phi = resolve_model_profile("microsoft/Phi-4-mini-instruct")
        qwen_profiles = [
            resolve_model_profile(f"Qwen/Qwen3-{size}")
            for size in ("0.6B", "1.7B", "4B", "8B")
        ]

        self.assertEqual(gemma3.loader, "multimodal")
        self.assertEqual(gemma3.prompt_format, "chat")
        self.assertTrue(phi.trust_remote_code)
        self.assertTrue(
            all(profile.enable_thinking is False for profile in qwen_profiles)
        )
        self.assertTrue(
            all(profile.prompt_format == "chat" for profile in qwen_profiles)
        )
        self.assertEqual(
            resolve_model_profile("Qwen/Qwen3-4B-Base").prompt_format,
            "plain",
        )
        self.assertEqual(
            resolve_model_profile("google/gemma-3-4b-pt").prompt_format,
            "plain",
        )

    def test_last_nonpadding_indices_support_both_padding_sides(self):
        mask = torch.tensor(
            [
                [1, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
            ]
        )
        self.assertTrue(
            torch.equal(
                last_nonpadding_indices(mask),
                torch.tensor([2, 3, 3]),
            )
        )
        with self.assertRaises(ValueError):
            last_nonpadding_indices(torch.zeros((1, 3), dtype=torch.long))

    def test_chat_tokenization_and_control_slot(self):
        tokenizer = DummyTokenizer()
        attach_model_profile(
            DummyModel(),
            tokenizer,
            ModelProfile(
                family="gemma4",
                loader="multimodal",
                prompt_format="chat",
                enable_thinking=False,
            ),
        )

        tokens = tokenize_prompts(tokenizer, "question", return_tensors="pt")
        prefix, suffix = prompt_control_parts(tokenizer, "question", max_length=256)

        self.assertNotEqual(int(tokens["input_ids"][0, 0]), 99)
        self.assertGreater(prefix.numel(), 0)
        self.assertGreater(suffix.numel(), 0)

    def test_nested_multimodal_layers_are_discovered(self):
        outer = torch.nn.Module()
        outer.model = torch.nn.Module()
        outer.model.language_model = torch.nn.Module()
        expected = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        outer.model.language_model.layers = expected

        self.assertIs(get_layers(outer), expected)

    def test_multi_layer_collection_reuses_each_forward_pass(self):
        class LayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(128, 4)
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(
                    [torch.nn.Identity(), torch.nn.Identity()]
                )
                self.forward_calls = 0

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, attention_mask=None):
                self.forward_calls += 1
                hidden = self.embedding(input_ids)
                for layer in self.model.layers:
                    hidden = layer(hidden)
                return type("Output", (), {"last_hidden_state": hidden})

        model = LayerModel()
        states = collect_residual_states_many(
            model,
            [0, 1],
            DummyTokenizer(),
            ["a", "longer", "third"],
            batch_size=2,
        )

        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(states[0].shape, (3, 4))
        self.assertEqual(states[1].shape, (3, 4))

    @mock.patch("llm_controllability.models.loading.transformers.AutoTokenizer.from_pretrained")
    @mock.patch("llm_controllability.models.loading.transformers.AutoModelForMultimodalLM.from_pretrained")
    def test_gemma_loader_uses_multimodal_class(self, model_loader, tokenizer_loader):
        model_loader.return_value = DummyModel()
        tokenizer_loader.return_value = DummyTokenizer()

        model, tokenizer = load_model(
            model_name="google/gemma-3-4b-it",
            device_map="cpu",
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )

        model_loader.assert_called_once()
        self.assertEqual(model_profile(model).family, "gemma3")
        self.assertEqual(model_profile(tokenizer).prompt_format, "chat")

    @mock.patch(
        "llm_controllability.models.runtime.platform.mac_ver",
        return_value=("14.6", ("", "", ""), ""),
    )
    @mock.patch(
        "llm_controllability.models.runtime.mps_is_available",
        return_value=True,
    )
    @mock.patch(
        "llm_controllability.models.runtime.torch.cuda.is_available",
        return_value=False,
    )
    @mock.patch(
        "llm_controllability.models.loading.transformers.AutoTokenizer.from_pretrained"
    )
    def test_loader_resolves_auto_to_mps(
        self,
        tokenizer_loader,
        _cuda,
        _mps,
        _mac,
    ):
        model_class = mock.Mock()
        model_class.from_pretrained.return_value = DummyModel()
        tokenizer_loader.return_value = DummyTokenizer()

        with mock.patch.dict(
            loading.transformers.__dict__,
            {"AutoModelForCausalLM": model_class},
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, _ = load_model(
                model_name="microsoft/phi-4",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

        self.assertTrue(any("flash_attention_2" in str(item.message) for item in caught))
        kwargs = model_class.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], "mps")
        self.assertEqual(kwargs["dtype"], torch.bfloat16)
        self.assertEqual(kwargs["attn_implementation"], "eager")
        runtime = model_runtime_config(model)
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.device, torch.device("mps"))

    def test_scope_identifiers_and_analysis(self):
        self.assertEqual(
            gemma_scope_release("google/gemma-3-4b-it"),
            "gemma-scope-2-4b-it-res-all",
        )
        self.assertEqual(
            gemma_scope_sae_id(12),
            "layer_12_width_16k_l0_small",
        )
        samples = []
        for example_id, baseline in (("a", [1.0, 0.0]), ("b", [0.0, 1.0])):
            samples.extend(
                [
                    _sample(
                        example_id,
                        ControlChannel.BASELINE,
                        "baseline",
                        baseline,
                    ),
                    _sample(
                        example_id,
                        ControlChannel.PROMPT,
                        "prompt",
                        np.asarray(baseline) + np.asarray([0.2, 0.1]),
                    ),
                    _sample(
                        example_id,
                        ControlChannel.ACTIVATION,
                        "activation",
                        np.asarray(baseline) + np.asarray([0.1, 0.2]),
                    ),
                ]
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            states = root / "states"
            output = root / "scope"
            save_state_samples(samples, states)
            manifest = run_gemma_scope_study(
                states,
                output,
                model_name="google/gemma-3-4b-it",
                layer=1,
                device="cpu",
                top_k=4,
                analysis_features=4,
                max_samples=None,
                sae=FakeSAE(),
            )

            geometry = (output / "feature_geometry.json").read_text(encoding="utf-8")

        self.assertEqual(manifest["n_samples"], 6)
        self.assertIn("prompt_activation_overlap", geometry)


if __name__ == "__main__":
    unittest.main()
