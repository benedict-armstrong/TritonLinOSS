"""
Pytest-based correctness tests for TritonLinOSS implementation
"""

import pytest
import torch

from ..models.TorchLinOSS import LinOSS as LinOSSTorch

SEED = 42


@pytest.mark.parametrize("layer_name", ["IM", "IMEX", "Damped"])
@pytest.mark.parametrize(
    "batch_size,seq_length,input_dim,hidden_dim,ssm_size,output_dim",
    [
        (2, 64, 8, 32, 64, 10),  # small
        (4, 256, 16, 64, 128, 20),  # medium
        (8, 512, 32, 128, 256, 40),  # large
    ],
    ids=["small", "medium", "large"],
)
def test_onnx_export(
    layer_name: str,
    batch_size: int,
    seq_length: int,
    input_dim: int,
    hidden_dim: int,
    ssm_size: int,
    output_dim: int,
):
    """ """

    model_params = {
        "layer_name": layer_name,
        "input_dim": input_dim,
        "state_dim": ssm_size,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_blocks": 2,
        "classification": True,
        "tanh_output": False,
        "output_step": 1,
    }

    test_input = torch.randn(batch_size, seq_length, input_dim)
    torch_model = LinOSSTorch(**model_params)

    # Export the model to ONNX
    exported_model = torch.export.export(
        torch_model,
        (test_input,),
    )

    torch.onnx.verification.verify_onnx_program(exported_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
