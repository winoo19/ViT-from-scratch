# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.utils import set_seed
from src.models import (
    LayerNormalization,
    MultiHeadAttention,
    AttentionEncoder,
)


@pytest.mark.order(7)
def test_layer_normalization() -> None:
    set_seed(42)
    # define inputs
    inputs: torch.Tensor = torch.rand(64, 30)
    inputs_torch: torch.Tensor = inputs.clone()

    # define layer normalization
    layer_norm = LayerNormalization(5, 6, eps=1e-6)
    layer_norm_torch = torch.nn.LayerNorm(30, eps=1e-6)

    # compute outputs
    outputs: torch.Tensor = layer_norm(inputs.view(64, 5, 6))
    outputs_torch: torch.Tensor = layer_norm_torch(inputs_torch)

    # check output type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect type, expected torch.Tensor got {type(outputs)}"

    # check output size
    assert (
        outputs.shape == inputs.shape
    ), f"Incorrect shape, expected {inputs.shape}, got {outputs.shape}"

    # check outputs of layer normalization
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, (
        "Incorrect outputs, outputs are not equal to pytorch implementation"
    )

    return None


@pytest.mark.order(8)
def test_multi_head_attention() -> None:
    set_seed(42)
    # define inputs
    inputs: torch.Tensor = torch.rand(64, 30, 512)
    inputs_torch: torch.Tensor = inputs.clone()

    # define multi head attention
    multi_head_attention = MultiHeadAttention(8, 512)
    multi_head_attention_torch = torch.nn.MultiheadAttention(512, 8)

    # compute outputs
    outputs: torch.Tensor = multi_head_attention(inputs)
    outputs_torch: torch.Tensor = multi_head_attention_torch(
        inputs_torch, inputs_torch, inputs_torch
    )[0]

    # check output type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect type, expected torch.Tensor got {type(outputs)}"

    # check output size
    assert (
        outputs.shape == inputs.shape
    ), f"Incorrect shape, expected {inputs.shape}, got {outputs.shape}"

    # check outputs of multi head attention
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, (
        "Incorrect outputs, outputs are not equal to pytorch implementation"
    )

    return None


@pytest.mark.order(9)
def test_attention_encoder() -> None:
    set_seed(42)
    # define inputs
    inputs: torch.Tensor = torch.rand(64, 30, 512)
    inputs_torch: torch.Tensor = inputs.clone()

    # define attention encoder
    attention_encoder = AttentionEncoder(512, 8)
    attention_encoder_torch = torch.nn.TransformerEncoderLayer(512, 8)

    # compute outputs
    outputs: torch.Tensor = attention_encoder(inputs)
    outputs_torch: torch.Tensor = attention_encoder_torch(inputs_torch)

    # check output type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect type, expected torch.Tensor got {type(outputs)}"

    # check output size
    assert (
        outputs.shape == inputs.shape
    ), f"Incorrect shape, expected {inputs.shape}, got {outputs.shape}"

    # check outputs of attention encoder
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, (
        "Incorrect outputs, outputs are not equal to pytorch implementation"
    )

    return None
