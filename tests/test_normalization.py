import torch

from model import RMSNorm


def test_rms_norm_preserves_shape():
    layer = RMSNorm(d_model=4)
    x = torch.randn(2, 3, 4)

    assert layer(x).shape == x.shape


def test_rms_norm_normalizes_last_dimension():
    layer = RMSNorm(d_model=2, eps=0.0)
    x = torch.tensor([[3.0, 4.0]])

    assert torch.allclose(layer(x), torch.tensor([[0.84852814, 1.13137085]]))


def test_rms_norm_applies_weight():
    layer = RMSNorm(d_model=2, eps=0.0)
    layer.weight.data = torch.tensor([2.0, 3.0])
    x = torch.tensor([[3.0, 4.0]])

    assert torch.allclose(layer(x), torch.tensor([[1.69705629, 3.39411259]]))


def test_rms_norm_weight_receives_gradient():
    layer = RMSNorm(d_model=3)
    x = torch.randn(2, 3)

    layer(x).sum().backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape


def test_rms_norm_normalizes_each_token_independently():
    layer = RMSNorm(d_model=2, eps=0.0)
    x = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])

    expected = torch.tensor([[[0.84852814, 1.13137085], [0.54392827, 1.30542785]]])

    assert torch.allclose(layer(x), expected)
