import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

### DELETED: selective_state_update is not used in this experiment since it does not support the use of timevariant dt, B, C flags.
# from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn
# try:
#     from mamba_ssm.ops.triton.selective_state_update import selective_state_update
# except ImportError:
#     selective_state_update = None



class Mamba_TimeVariant(nn.Module):
    """
    Mamba Block with support for time-variant dt, B, C.
    The time-variant parameters are controlled by `timevariant_dt`, `timevariant_B`, and `timevariant_C` flags.
    
    Difference from the original `modules.mamba_simple.Mamba` class:
    - In `step()`, `x_proj` can be `None`, so dt, B, and C are split only when guarded by if `self.x_proj` is not `None`.
    - When `tv_dt=False`, dt is constructed as a bias-based constant and expanded to shape `(B, d_inner)` via repeat to match einsum dimensions.
    - When `d_conv=0`, step avoids accessing depthwise convolution weights and instead follows the `SiLU(x)` path.
    - In cache creation (`allocate_inference_cache`, `_get_states_from_cache`), dtype and device are selected safely even when `conv1d` is `Identity`.
    """
    def __init__(
        self,
        d_model,
        d_input=None,   ### added
        d_output=None,  ### added
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,  # Fused kernel options. ** Fixed to False for this experiment **
        layer_idx=None,
        device=None,
        dtype=None,
        timevariant_dt=True,  ### ADDED: to support timevariant dt
        timevariant_B=True,   ### ADDED: to support timevariant B
        timevariant_C=True,   ### ADDED: to support timevariant C
        use_D=True,           ### ADDED: to control the usage of D parameter
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input if d_input is not None else d_model    ### ADDED: for various input dimensions
        self.d_output = d_output if d_output is not None else d_model ### ADDED: for various output dimensions
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        ### MODIFIED: change the in_feature dimension from d_model to d_input
        self.in_proj = nn.Linear(self.d_input, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        ) if d_conv > 0 else nn.Identity()  ### MODIFIED: Skip the convolution if d_conv is set to 0

        self.activation = "silu"
        self.act = nn.SiLU()

        ### MODIFIED: adjust the x_proj layer to support timevariant dt, B, C. 
        ###           this is possible since selective_scan.cpp has `is_variable_B` and `is_variable_C` flags that control the usage of timevariant B and C
        self.tv_dt, self.tv_B, self.tv_C = timevariant_dt, timevariant_B, timevariant_C
        self.tv_proj_dim = [0, 0, 0,]
        if timevariant_dt | timevariant_B | timevariant_C:
            if timevariant_dt:
                self.tv_proj_dim[0] = self.dt_rank
            if timevariant_B:
                self.tv_proj_dim[1] = self.d_state
            if timevariant_C:
                self.tv_proj_dim[2] = self.d_state
        self.x_proj = nn.Linear(
            self.d_inner, sum(self.tv_proj_dim), bias=False, **factory_kwargs
        ) if sum(self.tv_proj_dim) > 0 else None

        ### ADDED: if tv flags are False, we will use constants for dt, B, C
        if not timevariant_B:
            self.B = nn.Parameter(torch.rand(self.d_inner, self.d_state, **factory_kwargs))
            self.B._no_weight_decay = True
        if not timevariant_C:
            self.C = nn.Parameter(torch.rand(self.d_inner, self.d_state, **factory_kwargs))
            self.C._no_weight_decay = True

        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # (expand * d_model, d_state)
        # A = [[1, 2, ..., d_state], [1, 2, ..., d_state], ..., [1, 2, ..., d_state]]
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        ### MODIFIED: D is a learnable parameter only if use_D is True else it is not used
        ###         this is possible since selective_scan.cpp allows D to be optional
        if use_D:
            self.D = nn.Parameter(torch.ones(self.d_inner, device=device)).float()
            self.D._no_weight_decay = True
        else:
            self.D = None

        ### MODIFIED: out_proj now has d_output instead of d_model
        self.out_proj = nn.Linear(self.d_inner, self.d_output, bias=bias, **factory_kwargs)


    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, d_input = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )  # (d_inner * 2, d_input) @ (d_input, batch * seqlen) -> (d_inner * 2, batch, seqlen) -> (batch, d_inner * 2, seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state). always have negative values. 
        ### DELETED: Actually not used in this experiment since we should control the usage of timevariant dt,B,C
        # # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
        #     out = mamba_inner_fn(
        #         xz,
        #         self.conv1d.weight,
        #         self.conv1d.bias,
        #         self.x_proj.weight,
        #         self.dt_proj.weight,
        #         self.out_proj.weight,
        #         self.out_proj.bias,
        #         A,
        #         None,  # input-dependent B
        #         None,  # input-dependent C
        #         self.D,
        #         delta_bias=self.dt_proj.bias.float(),
        #         delta_softplus=True,
        #     )
        # else:
        x, z = xz.chunk(2, dim=1)  # (batch, d_inner, seqlen), (batch, d_inner, seqlen)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            if self.d_conv > 0:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        
        ### MODIFIED: use causal_conv if available
        if (causal_conv1d_fn is None) or (self.d_conv not in [2, 3, 4]):
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )  # (batch, d_inner, seqlen)

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        ### MODIFIED: x_proj is now optional and only used if either timevariant dt, B, or C is True
        if self.x_proj is not None:
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (batch, d_inner, seqlen) -> (batch * seqlen, d_inner) -> (batch * seqlen, ...) depending on timevariant flags
            dt, B, C = torch.split(x_dbl, self.tv_proj_dim, dim=-1)  # (batch * seqlen, dt_rank), (batch * seqlen, d_state), (batch * seqlen, d_state) if enabled
        
        ### MODIFIED: dt, B, C are now set based on each timevariant flags
        # If timevariant dt is False, we use a constant dt, which will be set in delta_bias parameter. Thus, we don't need to compute dt here.
        if not self.tv_dt:
            dt = torch.zeros(batch, self.d_inner, seqlen, device=self.dt_proj.bias.device, dtype=self.dt_proj.bias.dtype)  # (batch, d_inner, seqlen)
        else:
            dt = self.dt_proj.weight @ dt.t()  # (d_inner, d_rank) @ (d_rank, batch * seqlen) -> (d_inner, batch * seqlen)
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)  # (batch, d_inner, seqlen)
        # if timevariant B is False, we use a constant B, which is defined in __init__.
        if not self.tv_B:
            B = self.B  # (d_inner, d_state)
        else:
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # (b, dstate, l)
        # if timevariant C is False, we use a constant C, which is defined in __init__.
        if not self.tv_C:
            C = self.C  # (d_inner, d_state)
        else:
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # (b, dstate, l)


        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D,
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (batch, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # (batch, d_inner), (batch, d_inner)

        # Conv step
        if self.d_conv == 0:
            x = self.act(x).to(dtype=dtype)
        elif (causal_conv1d_update is None) or (self.d_conv not in [2, 3, 4]):
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        if self.x_proj is not None:
            x_db = self.x_proj(x)  # (B, d_inner) -> (B, dt_rank + d_state + d_state)
            dt, B, C = torch.split(x_db, self.tv_proj_dim, dim=-1)  # (B, dt_rank), (B, d_state), (B, d_state)

        # SSM step
        ### DELETED: selective_state_update function does not support the use of timevariant dt, B, C.
        # if selective_state_update is None:
        #     dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        #     dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        #     dB = torch.einsum("bd,bn->bdn", dt, B)
        #     ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        #     y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        #     y = y + self.D.to(dtype) * x
        #     y = y * self.act(z)  # (B D)
        # else:
        #     y = selective_state_update(
        #         ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
        #     )

        ### MODIFIED: dt, B are now set based on the timevariant flags.
        if not self.tv_dt:
            dt = F.softplus(self.dt_proj.bias.to(dtype=x.dtype))
            dt = repeat(dt, "d -> b d", b=x.shape[0])  # (B, d_inner)
        else:
            dt = F.linear(dt, self.dt_proj.weight)  # (B, dt_rank) @ (dt_rank, d_inner) -> (B, d_inner)
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))  # (B, d_inner)
        
        if not self.tv_B:
            dB = torch.einsum("bd,dn->bdn", dt, self.B)  # (B, d_inner, d_state)
        else:
            dB = torch.einsum("bd,bn->bdn", dt, B)  # (B, d_inner, d_state)
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)  # (B, d_inner, d_state)
        
        ### MODIFIED: C is now set based on the timevariant flags.
        if not self.tv_C:
            y = torch.einsum("bdn,dn->bd", ssm_state.to(dtype), self.C)  # (B, d_inner, d_state) @ (d_inner, d_state) -> (B, d_inner)
        else:
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)  # (B, d_inner, d_state) @ (B, d_state) -> (B, d_inner)

        ### MODIFIED: skip connection is now applied based on the use_D flag.
        if self.D is not None:
            y = y + self.D.to(dtype) * x  # (B, d_inner) + (d_inner) * (B, d_inner) -> (B, d_inner)

        y = y * self.act(z)  # (B, d_inner)

        out = self.out_proj(y)  # (B, d_inner) -> (B, d_output)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = (self.conv1d.weight.dtype if hasattr(self.conv1d, "weight") else self.in_proj.weight.dtype) if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=(self.conv1d.weight.device if hasattr(self.conv1d, "weight") else self.in_proj.weight.device),
                dtype=(self.conv1d.weight.dtype if hasattr(self.conv1d, "weight") else self.in_proj.weight.dtype),
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state