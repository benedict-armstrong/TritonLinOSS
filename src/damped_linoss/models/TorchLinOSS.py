import abc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from typing import Tuple


try:
    from ..parallel_scan.torch_interface import (
        ParallelScanFunction,
        TRITON_AVAILABLE,
    )
except ImportError:
    TRITON_AVAILABLE = False
    ParallelScanFunction = None

from ..parallel_scan.torch_associative_scan import associative_scan


SCAN_TYPE = Tuple[torch.Tensor, torch.Tensor]


def binary_operator(q_i: SCAN_TYPE, q_j: SCAN_TYPE) -> SCAN_TYPE:
    """
    Binary operator for parallel scan of linear recurrence.

    Args:
        q_i: tuple (A_i, b_i)
             A_i shape: (..., 4 * P)
             b_i shape: (..., 2 * P, 2)
        q_j: tuple (A_j, b_j)
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    iA, iB, iC, iD = torch.chunk(A_i, 4, dim=-1)
    jA, jB, jC, jD = torch.chunk(A_j, 4, dim=-1)

    A_new_part = jA * iA + jB * iC
    B_new_part = jA * iB + jB * iD
    C_new_part = jC * iA + jD * iC
    D_new_part = jC * iB + jD * iD

    # Concatenate back to shape (..., 4*P)
    A_new = torch.cat([A_new_part, B_new_part, C_new_part, D_new_part], dim=-1)

    b_i1, b_i2 = torch.chunk(b_i, 2, dim=-2)

    # unsqueeze the last dim of A parts to broadcast over the complex dimension of b.
    jA_bs = jA.unsqueeze(-1)
    jB_bs = jB.unsqueeze(-1)
    jC_bs = jC.unsqueeze(-1)
    jD_bs = jD.unsqueeze(-1)

    new_b1 = jA_bs * b_i1 + jB_bs * b_i2
    new_b2 = jC_bs * b_i1 + jD_bs * b_i2

    new_b = torch.cat([new_b1, new_b2], dim=-2)

    return A_new, new_b + b_j


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initializes the Gated Linear Unit (GLU) module.
        """
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        """
        Applies the GLU activation: w1(x) * sigmoid(w2(x))
        Handles (L, H) and (B, L, H) inputs.
        """
        return self.w1(x) * torch.sigmoid(self.w2(x))


class _AbstractLinOSSLayer(nn.Module):
    def __init__(self, use_triton=None):
        super().__init__()
        self.use_triton = use_triton

    @abc.abstractmethod
    def _recurrence(self):
        raise NotImplementedError

    def _should_use_triton(self, tensor):
        """Determine whether to use Triton backend based on override flag and tensor device."""
        # if self.use_triton is None:
        #     # Auto: use Triton if available and tensor is on CUDA
        #     use_triton = TRITON_AVAILABLE and tensor.is_cuda

        #     # Warn if CUDA is available but Triton is not
        #     if tensor.is_cuda and not TRITON_AVAILABLE:
        #         warnings.warn(
        #             "Running on CUDA but Triton is not available. "
        #             "Performance may be suboptimal. "
        #             "Install Triton with 'pip install damped-linoss[cuda]' for better performance.",
        #             UserWarning,
        #             stacklevel=4,
        #         )

        #     # Warn if Triton is available but tensor is not on CUDA
        #     if TRITON_AVAILABLE and not tensor.is_cuda:
        #         warnings.warn(
        #             "Triton is available but tensor is not on CUDA. "
        #             "Falling back to PyTorch native backend. "
        #             "Move tensors to CUDA for better performance.",
        #             UserWarning,
        #             stacklevel=4,
        #         )

        #     return use_triton
        # else:
        #     # Manual override
        #     if self.use_triton and not TRITON_AVAILABLE:
        #         raise RuntimeError(
        #             "Triton backend requested but not available. "
        #             "Install with 'pip install damped-linoss[cuda]'."
        #         )
        #     return self.use_triton
        return True


class IMLayer(_AbstractLinOSSLayer):
    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, use_triton=None):
        super().__init__(use_triton)

        # Initialize parameters
        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)

        self.A_diag = nn.Parameter(torch.empty(state_dim))
        init.uniform_(self.A_diag, 0.0, 1.0)  # Matches jax.random.uniform default

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / torch.sqrt(torch.tensor(state_dim, dtype=torch.float32))
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _recurrence(self, A_diag, B, input_sequence, step):
        """Compute the LxP output of LinOSS-IM given an LxH or BxLxH input.
        Args:
            A_diag          (float32):      diagonal state matrix     (P,)
            B               (float32):      input matrix              (P, H, 2)
            input_sequence  (float32):      input sequence            (L, H) or (B, L, H)
            step            (float):        discretization time-step  (P,)
        Returns:
            ys              (float32):      SSM states                (L, P, 2) or (B, L, P, 2)
        """
        # Use '...' to handle optional batch dim
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B)

        schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
        M_11 = 1.0 - step**2.0 * A_diag * schur_comp
        M_12 = -1.0 * step * A_diag * schur_comp
        M_21 = step * schur_comp
        M_22 = schur_comp

        M = torch.cat([M_11, M_12, M_21, M_22])

        L = input_sequence.shape[-2]

        # If batched, expand M to (B, 4*P)
        if input_sequence.dim() == 3:
            B = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(B, -1)  # (B, 4*P)
        else:
            M_elements = M  # (4*P,)

        # Reshape params for broadcasting with (..., L, P, 2)
        view_shape = (1, -1, 1)  # Shape for (L, P, 2)
        if input_sequence.dim() == 3:
            view_shape = (1, 1, -1, 1)  # Shape for (B, L, P, 2)

        M_11_b = M_11.view(view_shape)
        M_21_b = M_21.view(view_shape)
        step_b = step.view(view_shape)

        F1 = M_11_b * Bu_elements * step_b
        F2 = M_21_b * Bu_elements * step_b

        # F shape is (..., L, 2*P, 2)
        F = torch.cat((F1, F2), dim=-2)  # Concat on the P dim

        # M_elements: (..., 4*P), F: (..., L, 2*P, 2)
        P = A_diag.shape[0]

        if self._should_use_triton(input_sequence):
            _, xs = ParallelScanFunction.apply(M_elements, F)
        else:
            if input_sequence.dim() == 3:
                M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
                scan_axis = 1
            else:
                M_expanded = M_elements.unsqueeze(0).expand(L, -1)
                scan_axis = 0

            _, xs = associative_scan(
                binary_operator,
                (M_expanded, F),
                reverse=False,
                axis=scan_axis,
            )

        # Return (..., L, P, 2)
        return xs[..., A_diag.shape[0] :, :]

    def forward(self, input_sequence):
        # input_sequence is (L, H) or (B, L, H)
        steps = torch.sigmoid(self.steps)
        A_diag = F.relu(self.A_diag)

        # ys is (..., L, P, 2)
        ys = self._recurrence(A_diag, self.B, input_sequence, steps)

        # Apply SSM Output Operations Cx + Du
        # C is (H, P, 2)
        # Cy_einsum is (..., L, H, 2)
        Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
        Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
        Du = input_sequence * self.D
        xs = Cy + Du

        return xs


class IMEXLayer(_AbstractLinOSSLayer):
    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max, use_triton=None):
        super().__init__(use_triton)

        # Initialize parameters
        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)

        self.A_diag = nn.Parameter(torch.empty(state_dim))
        init.uniform_(self.A_diag, 0.0, 1.0)  # Matches jax.random.uniform default

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / torch.sqrt(torch.tensor(state_dim, dtype=torch.float32))
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _recurrence(self, A_diag, B, input_sequence, step):
        """Compute the LxP output of LinOSS-IM given an LxH or BxLxH input.
        Args:
            A_diag          (float32):      diagonal state matrix     (P,)
            B               (float32):      input matrix              (P, H, 2)
            input_sequence  (float32):      input sequence            (L, H) or (B, L, H)
            step            (float):        discretization time-step  (P,)
        Returns:
            ys              (float32):      SSM states                (L, P, 2) or (B, L, P, 2)
        """
        # Use '...' to handle optional batch dim
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B)

        M_11 = torch.ones_like(A_diag)
        M_12 = -1.0 * step * A_diag
        M_21 = step
        M_22 = 1.0 - (step**2.0) * A_diag

        M = torch.cat([M_11, M_12, M_21, M_22])

        L = input_sequence.shape[-2]

        # If batched, expand M to (B, 4*P)
        if input_sequence.dim() == 3:
            B = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(B, -1)  # (B, 4*P)
        else:
            M_elements = M  # (4*P,)

        # Reshape params for broadcasting with (..., L, P, 2)
        view_shape = (1, -1, 1)  # Shape for (L, P, 2)
        if input_sequence.dim() == 3:
            view_shape = (1, 1, -1, 1)  # Shape for (B, L, P, 2)

        step_b = step.view(view_shape)

        F1 = Bu_elements * step_b
        F2 = Bu_elements * (step_b**2.0)

        # F shape is (..., L, 2*P, 2)
        F = torch.cat((F1, F2), dim=-2)  # Concat on the P dim

        # M_elements: (..., 4*P), F: (..., L, 2*P, 2)
        P = A_diag.shape[0]

        if self._should_use_triton(input_sequence):
            if torch.onnx.is_in_onnx_export():
                (_, xs) = torch.onnx.ops.symbolic_multi_out(
                    "triton_kernels::ParallelScan",
                    (M_elements, F),
                    dtypes=(M_elements.dtype, F.dtype),
                    shapes=((B, L, 4 * P), (B, L, 2 * P, 2)),
                    version=1,
                )
            else:
                _, xs = ParallelScanFunction.apply(M_elements, F)
        else:
            if input_sequence.dim() == 3:
                M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
                scan_axis = 1
            else:
                M_expanded = M_elements.unsqueeze(0).expand(L, -1)
                scan_axis = 0

            _, xs = associative_scan(
                binary_operator,
                (M_expanded, F),
                reverse=False,
                axis=scan_axis,
            )

        # Return (..., L, P, 2)
        return xs[..., A_diag.shape[0] :, :]

    def forward(self, input_sequence):
        # input_sequence is (L, H) or (B, L, H)
        steps = torch.sigmoid(self.steps)
        A_diag = F.relu(self.A_diag)

        # ys is (..., L, P, 2)
        ys = self._recurrence(A_diag, self.B, input_sequence, steps)

        # Apply SSM Output Operations Cx + Du
        # C is (H, P, 2)
        # Cy_einsum is (..., L, H, 2)
        Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
        Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
        Du = input_sequence * self.D
        xs = Cy + Du

        return xs


class DampedLayer(_AbstractLinOSSLayer):
    def __init__(self, state_dim, hidden_dim, r_min, r_max, theta_max):
        super().__init__()

        self.steps = nn.Parameter(torch.empty(state_dim))
        init.normal_(self.steps, std=0.5)
        steps = torch.sigmoid(self.steps)

        mags = torch.sqrt(torch.rand(state_dim) * (r_max**2 - r_min**2) + r_min**2)
        G_diag_init = (1 - mags**2) / (steps.detach() * mags**2)
        self.G_diag = nn.Parameter(G_diag_init)

        theta = torch.rand(state_dim) * theta_max
        A_diag_init = self._map_theta_to_A(
            theta, F.relu(self.G_diag.detach()), steps.detach()
        )
        self.A_diag = nn.Parameter(A_diag_init)

        self.B = nn.Parameter(torch.empty(state_dim, hidden_dim, 2))
        std_b = 1.0 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        init.uniform_(self.B, -std_b, std_b)

        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim, 2))
        std_c = 1.0 / torch.sqrt(torch.tensor(state_dim, dtype=torch.float32))
        init.uniform_(self.C, -std_c, std_c)

        self.D = nn.Parameter(torch.empty(hidden_dim))
        init.normal_(self.D, std=1.0)

    def _map_theta_to_A(self, thetas, G_diag, steps):
        """Map theta angles to A diagonal values."""
        cos_theta = torch.cos(thetas)
        tan_theta = torch.tan(thetas)
        tan_theta_sq = tan_theta**2

        sqrt_term = 4 * torch.sqrt(
            steps**4 * cos_theta ** (-2) + steps**5 * G_diag * cos_theta ** (-2)
        )

        common_term = steps**2 * (
            -4
            - 2 * steps * G_diag
            - 4 * tan_theta_sq
            - 2 * steps * G_diag * tan_theta_sq
        )

        denominator = 2 * steps**4 * (1 + tan_theta_sq)

        A_plus = (sqrt_term - common_term) / denominator
        A_minus = (-sqrt_term - common_term) / denominator

        A_diag = torch.where(thetas > math.pi / 2, A_plus, A_minus)

        return A_diag

    def _recurrence(self, A_diag, G_diag, B, input_sequence, step):
        """Compute the LxP output of Damped-LinOSS given an LxH or BxLxH input.
        Args:
            A_diag          (float32):      diagonal state matrix     (P,)
            G_diag          (float32):      diagonal damping matrix   (P,)
            B               (float32):      input matrix              (P, H, 2)
            input_sequence  (float32):      input sequence            (L, H) or (B, L, H)
            step            (float):        discretization time-step  (P,)
        Returns:
            ys              (float32):      SSM states                (L, P, 2) or (B, L, P, 2)
        """
        # Use '...' to handle optional batch dim
        Bu_elements = torch.einsum("...lh,pht->...lpt", input_sequence, B)

        Identity = torch.ones_like(A_diag)
        S = Identity + step * G_diag
        M_11 = 1.0 / S
        M_12 = -step / S * A_diag
        M_21 = step / S
        M_22 = Identity - step**2 / S * A_diag

        M = torch.cat([M_11, M_12, M_21, M_22])

        L = input_sequence.shape[-2]

        # If batched, expand M to (B, 4*P)
        if input_sequence.dim() == 3:
            B = input_sequence.shape[0]
            M_elements = M.unsqueeze(0).expand(B, -1)  # (B, 4*P)
        else:
            M_elements = M  # (4*P,)

        # Reshape params for broadcasting with (..., L, P, 2)
        view_shape = (1, -1, 1)  # Shape for (L, P, 2)
        if input_sequence.dim() == 3:
            view_shape = (1, 1, -1, 1)  # Shape for (B, L, P, 2)

        step_b = step.view(view_shape)
        S_b = S.view(view_shape)

        F1 = step_b * (1.0 / S_b) * Bu_elements
        F2 = (step_b**2) * (1.0 / S_b) * Bu_elements

        # F shape is (..., L, 2*P, 2)
        F = torch.cat((F1, F2), dim=-2)  # Concat on the P dim

        # M_elements: (..., 4*P), F: (..., L, 2*P, 2)
        P = A_diag.shape[0]

        if self._should_use_triton(input_sequence):
            _, xs = ParallelScanFunction.apply(M_elements, F)
        else:
            if input_sequence.dim() == 3:
                M_expanded = M_elements.unsqueeze(1).expand(-1, L, -1)
                scan_axis = 1
            else:
                M_expanded = M_elements.unsqueeze(0).expand(L, -1)
                scan_axis = 0

            _, xs = associative_scan(
                binary_operator,
                (M_expanded, F),
                reverse=False,
                axis=scan_axis,
            )

        # Return (..., L, P, 2)
        return xs[..., A_diag.shape[0] :, :]

    def forward(self, input_sequence):
        # input_sequence is (L, H) or (B, L, H)
        steps = torch.sigmoid(self.steps)
        G_diag = F.relu(self.G_diag)

        # ys is (..., L, P, 2)
        ys = self._recurrence(self.A_diag, G_diag, self.B, input_sequence, steps)

        # Apply SSM Output Operations Cx + Du
        # C is (H, P, 2)
        # Cy_einsum is (..., L, H, 2)
        Cy_complex = torch.einsum("...lpt,hpt->...lht", ys, self.C)
        Cy = Cy_complex[..., 0] - Cy_complex[..., 1]
        Du = input_sequence * self.D
        xs = Cy + Du

        return xs


class LinOSSBlock(nn.Module):
    def __init__(
        self,
        layer_name,
        state_dim,
        hidden_dim,
        r_min,
        r_max,
        theta_max,
        drop_rate,
    ):
        super().__init__()
        layer_map = {
            "IM": IMLayer,
            "IMEX": IMEXLayer,
            "Damped": DampedLayer,
        }
        if layer_name not in layer_map.keys():
            raise KeyError(f"Layer name {layer_name} not defined.")

        # BatchNorm1d *requires* (N, C, L)
        # Our C (Channels) is hidden_dim
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False)

        self.layer = layer_map[layer_name](
            state_dim,
            hidden_dim,
            r_min,
            r_max,
            theta_max,
        )
        self.glu = GLU(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # x shape is (L, H) or (B, L, H)
        skip = x

        # Apply BatchNorm
        # We need (N, C, L) where C = hidden_dim
        x_t = x
        if x.dim() == 2:
            x_t = x_t.unsqueeze(0)
        x_norm = self.norm(x_t.permute(0, 2, 1)).permute(0, 2, 1)
        if x.dim() == 2:
            x_norm = x_norm.squeeze(0)
        x = x_norm

        x = self.layer(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.glu(x)
        x = self.drop(x)
        x = skip + x

        return x


class LinOSS(nn.Module):
    def __init__(
        self,
        layer_name,
        input_dim,
        state_dim,
        hidden_dim,
        output_dim,
        num_blocks,
        classification=False,
        tanh_output=False,
        output_step=1,
        r_min=0.9,
        r_max=1.0,
        theta_max=math.pi,
        drop_rate=0.05,
        **kwargs,
    ):
        super().__init__()

        self.linear_encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LinOSSBlock(
                    layer_name,
                    state_dim,
                    hidden_dim,
                    r_min,
                    r_max,
                    theta_max,
                    drop_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_decoder = nn.Linear(hidden_dim, output_dim)

        self.classification = classification
        self.tanh_output = tanh_output
        self.output_step = output_step

    def forward(self, x):
        # x is (L, H_in) or (B, L, H_in)
        x = self.linear_encoder(x)

        for block in self.blocks:
            x = block(x)

        if self.classification:
            x = torch.mean(x, dim=-2)
            x = self.linear_decoder(x)
            x = F.softmax(x, dim=-1)
        else:
            x = x[..., self.output_step - 1 :: self.output_step, :]
            x = self.linear_decoder(x)
            if self.tanh_output:
                x = torch.tanh(x)

        return x
