import argparse
import os
from uuid import uuid4
from onnxruntime_extensions import PyCustomOpDef, onnx_op

import torch
from ..models.TorchLinOSS import LinOSS as LinOSSTorch
from ..parallel_scan.torch_interface import ParallelScanFunction


def export(
    output_path: str = "model.onnx",
    device: str | None = None,
):
    model_params = {
        "layer_name": "IMEX",
        "input_dim": 4,
        "state_dim": 4,
        "hidden_dim": 16,
        "output_dim": 4,
        "num_blocks": 2,
        "classification": True,
        "tanh_output": False,
        "output_step": 1,
    }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_input = torch.randn(10, 10, 4).to(device)
    torch_model = LinOSSTorch(**model_params).to(device)

    torch_model = torch_model.eval()

    # (M_elements, F),
    # dtypes=(M_elements.dtype, F.dtype),
    # shapes=((B, L, 4 * P), (B, L, 2 * P, 2)),

    # @onnx_op(
    #     op_type="triton_kernels::ParallelScan",
    #     inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    #     outputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    # )
    from torch.library import custom_op

    @custom_op("triton_kernels::ParallelScan", mutates_args=())
    def _model_forward(
        M: torch.Tensor, F: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return ParallelScanFunction.apply(M, F)

    # Ensure gradients are not tracked
    with torch.no_grad():
        # Export the model to ONNX
        torch.onnx.export(
            torch_model,
            (test_input,),
            f=output_path,
            report=True,
            verify=True,
            opset_version=20,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    job_id = os.getenv("CONDOR_JOB_ID", default=uuid4().hex[:8])

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to run on [default to GPU if available, else CPU]",
    )

    args = parser.parse_args()

    export(
        output_path=f"model_{job_id}.onnx",
        device=args.device,
    )
