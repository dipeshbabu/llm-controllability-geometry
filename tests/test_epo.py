import unittest
from typing import ClassVar

import numpy as np
import torch
from torch import nn

from llm_controllability.models.adapters import ensure_padding_token
from llm_controllability.optimization.epo import (
    History,
    build_pareto_frontier,
    combine_score,
    epo,
)
from llm_controllability.optimization.runners import residual_runner


class DummyPadTokenizer:
    pad_token = None
    eos_token = "<eos>"


class DummyTokenizer:
    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


class TinyTokenizer:
    vocab_size = 8
    all_special_ids: ClassVar[list[int]] = [0, 7]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 8)
        self.device = torch.device("cpu")

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None):
        hidden = self.embed(input_ids) if input_ids is not None else inputs_embeds
        return type("Output", (), {"logits": self.proj(hidden)})


def tiny_runner(input_ids=None, inputs_embeds=None):
    out = tiny_runner.model(input_ids=input_ids, inputs_embeds=inputs_embeds)
    return {"logits": out.logits, "target": out.logits[:, -1, 0]}


class EpoObjectiveTests(unittest.TestCase):
    def test_ensure_padding_token_uses_eos_when_available(self):
        tokenizer = DummyPadTokenizer()

        tokenizer, added = ensure_padding_token(tokenizer)

        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertFalse(added)

    def test_combine_score_prefers_lower_target_in_minimize_mode(self):
        target = torch.tensor([1.0, -2.0])
        xentropy = torch.tensor([0.5, 0.5])
        penalties = torch.tensor([1.0])

        scores = combine_score(target, xentropy, penalties, minimize=True)

        self.assertGreater(scores[0, 1].item(), scores[0, 0].item())

    def test_pareto_frontier_supports_minimization(self):
        history = History()
        history.ids = np.array([[[1], [2]]])
        history.target = np.array([[10.0, -10.0]])
        history.xentropy = np.array([[0.0, 0.0]])
        history.keep = np.array([[0, 1]])
        history.runtime = np.array([0.0])

        frontier = build_pareto_frontier(
            DummyTokenizer(), history, Xvs=np.array([1.0]), minimize=True
        )

        self.assertEqual(frontier.target[0], -10.0)
        self.assertEqual(frontier.text[0], "2")

    def test_disabled_callback_runs_all_iterations_and_accepts_final_keyword(self):
        tiny_runner.model = TinyModel()

        history = epo(
            tiny_runner,
            tiny_runner.model,
            TinyTokenizer(),
            seq_len=3,
            population_size=2,
            iters=2,
            batch_size=2,
            topk=2,
            explore_per_pop=1,
            callback=False,
        )

        self.assertEqual(history.ids.shape[0], 2)
        self.assertFalse(np.isin(history.ids, TinyTokenizer.all_special_ids).any())

    def test_residual_runner_projects_the_post_block_state(self):
        class AddOne(nn.Module):
            def forward(self, hidden):
                return hidden + 1.0

        class ResidualModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(4, 2)
                nn.init.zeros_(self.embed.weight)
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([AddOne()])

            def get_input_embeddings(self):
                return self.embed

            def forward(self, input_ids=None, inputs_embeds=None):
                hidden = (
                    self.embed(input_ids)
                    if input_ids is not None
                    else inputs_embeds
                )
                hidden = self.model.layers[0](hidden)
                return type("Output", (), {"logits": hidden})

        model = ResidualModel()
        runner = residual_runner(
            model,
            TinyTokenizer(),
            layer=0,
            vector=torch.tensor([1.0, 0.0]),
        )
        result = runner(input_ids=torch.tensor([[1, 2]]))

        self.assertTrue(torch.allclose(result["target"], torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
