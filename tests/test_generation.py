import torch
from torch import Tensor, nn

from generation import generate


class _SequenceModel(nn.Module):
    def __init__(
        self, context_length: int, steps: list[list[int]], vocab_size: int = 8
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.steps = steps
        self.vocab_size = vocab_size
        self.inputs: list[Tensor] = []

    def forward(self, input_ids: Tensor) -> Tensor:
        self.inputs.append(input_ids.clone())
        batch_size, sequence_length = input_ids.shape
        step = min(len(self.inputs) - 1, len(self.steps) - 1)
        next_tokens = torch.tensor(self.steps[step], device=input_ids.device)
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[torch.arange(batch_size), -1, next_tokens] = 100.0
        return logits


class _ProbabilityModel(nn.Module):
    context_length = 4
    next_logits: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("next_logits", torch.tensor([0.55, 0.30, 0.15]).log())

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, sequence_length = input_ids.shape
        return self.next_logits.expand(batch_size, sequence_length, -1)


def test_generate_truncates_each_model_context_and_restores_mode():
    model = _SequenceModel(context_length=3, steps=[[4], [5]])
    model.train()
    input_ids = torch.tensor([[0, 1, 2, 3]])

    output = generate(model, input_ids, maximum_new_tokens=2)

    torch.testing.assert_close(output, torch.tensor([[0, 1, 2, 3, 4, 5]]))
    torch.testing.assert_close(model.inputs[0], torch.tensor([[1, 2, 3]]))
    torch.testing.assert_close(model.inputs[1], torch.tensor([[2, 3, 4]]))
    assert model.training


def test_generate_top_p_keeps_the_smallest_probability_nucleus():
    model = _ProbabilityModel()
    input_ids = torch.zeros((8, 1), dtype=torch.long)

    output = generate(model, input_ids, maximum_new_tokens=1, top_p=0.55)

    torch.testing.assert_close(output[:, -1], torch.zeros(8, dtype=torch.long))


def test_generate_stops_when_every_batch_row_reaches_eos():
    eos_token_id = 7
    model = _SequenceModel(
        context_length=4,
        steps=[[eos_token_id, 1], [2, eos_token_id], [3, 3]],
    )
    input_ids = torch.tensor([[0], [0]])

    output = generate(
        model,
        input_ids,
        maximum_new_tokens=5,
        eos_token_id=eos_token_id,
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[0, eos_token_id, eos_token_id], [0, 1, eos_token_id]]),
    )
    assert len(model.inputs) == 2
