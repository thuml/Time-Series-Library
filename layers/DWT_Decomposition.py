# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5
@author: Murad
SISLab, USF
mmurad@usf.edu
https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer
"""

import torch
import torch.nn as nn
import pywt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function


class Decomposition(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 no_decomposition=[],
                 use_amp=[]):
        super(Decomposition, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.device = device
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.eps = 1e-5

        self.dwt = DWT1DForward(wave=self.wavelet_name, J=self.level,
                                use_amp=self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DForward(
            wave=self.wavelet_name, J=self.level, use_amp=self.use_amp)
        self.idwt = DWT1DInverse(wave=self.wavelet_name,
                                 use_amp=self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DInverse(
            wave=self.wavelet_name, use_amp=self.use_amp)

        self.input_w_dim = self._dummy_forward(self.input_length) if not self.no_decomposition else [
            self.input_length]  # length of the input seq after decompose
        self.pred_w_dim = self._dummy_forward(self.pred_length) if not self.no_decomposition else [
            self.pred_length]  # required length of the pred seq after decom

        self.tfactor = tfactor
        self.dfactor = dfactor
        #################################
        self.affine = False
        #################################

        if self.affine:
            self._init_params()

    def transform(self, x):
        # input: x shape: batch, channel, seq
        if not self.no_decomposition:
            yl, yh = self._wavelet_decompose(x)
        else:
            yl, yh = x, []  # no decompose: returning the same value in yl
        return yl, yh

    def inv_transform(self, yl, yh):
        if not self.no_decomposition:
            x = self._wavelet_reverse_decompose(yl, yh)
        else:
            x = yl  # no decompose: returning the same value in x
        return x

    def _dummy_forward(self, input_length):
        dummy_x = torch.ones((self.batch_size, self.channel, input_length)).to(self.device)
        yl, yh = self.dwt(dummy_x)
        l = []
        l.append(yl.shape[-1])
        for i in range(len(yh)):
            l.append(yh[i].shape[-1])
        return l

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones((self.level + 1, self.channel)))
        self.affine_bias = nn.Parameter(torch.zeros((self.level + 1, self.channel)))

    def _wavelet_decompose(self, x):
        # input: x shape: batch, channel, seq
        yl, yh = self.dwt(x)

        if self.affine:
            yl = yl.transpose(1, 2)  # batch, seq, channel
            yl = yl * self.affine_weight[0]
            yl = yl + self.affine_bias[0]
            yl = yl.transpose(1, 2)  # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ * self.affine_weight[i + 1]
                yh_ = yh_ + self.affine_bias[i + 1]
                yh[i] = yh_.transpose(1, 2)  # batch, channel, seq

        return yl, yh

    def _wavelet_reverse_decompose(self, yl, yh):
        if self.affine:
            yl = yl.transpose(1, 2)  # batch, seq, channel
            yl = yl - self.affine_bias[0]
            yl = yl / (self.affine_weight[0] + self.eps)
            yl = yl.transpose(1, 2)  # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ - self.affine_bias[i + 1]
                yh_ = yh_ / (self.affine_weight[i + 1] + self.eps)
                yh[i] = yh_.transpose(1, 2)  # batch, channel, seq

        x = self.idwt((yl, yh))
        return x  # shape: batch, channel, seq


###############################################################################################
"""
Following codes are combined from https://github.com/fbcotter/pytorch_wavelets. 
To use Wavelet decomposition, you do not need to modify any of the codes below this line,
we can just play with the class Decomposition(above)
"""


###############################################################################################

class DWT1DForward(nn.Module):
    """ Performs a 1d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """

    def __init__(self, J=1, wave='db1', mode='zero', use_amp=False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
        else:
            assert len(wave) == 2
            h0, h1 = wave[0], wave[1]

        # Prepare the filters - this makes them into column filters
        filts = prep_filt_afb1d(h0, h1)
        self.register_buffer('h0', filts[0])
        self.register_buffer('h1', filts[1])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, L_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients.
        """
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        mode = mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            x0, x1 = AFB1D.apply(x0, self.h0, self.h1, mode, self.use_amp)
            highs.append(x1)

        return x0, highs


class DWT1DInverse(nn.Module):
    """ Performs a 1d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, wave='db1', mode='zero', use_amp=False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            assert len(wave) == 2
            g0, g1 = wave[0], wave[1]

        # Prepare the filters
        filts = prep_filt_sfb1d(g0, g1)
        self.register_buffer('g0', filts[0])
        self.register_buffer('g1', filts[1])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, should
              match the format returned by DWT1DForward.

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, L_{in})`

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = mode_to_int(self.mode)
        # Do a multilevel inverse transform
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)

            # 'Unpad' added signal
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            x0 = SFB1D.apply(x0, x1, self.g0, self.g1, mode, self.use_amp)
        return x0


def roll(x, n, dim, make_even=False):
    if n < 0:
        n = x.shape[dim] + n

    if make_even and x.shape[dim] % 2 == 1:
        end = 1
    else:
        end = 0

    if dim == 0:
        return torch.cat((x[-n:], x[:-n + end]), dim=0)
    elif dim == 1:
        return torch.cat((x[:, -n:], x[:, :-n + end]), dim=1)
    elif dim == 2 or dim == -2:
        return torch.cat((x[:, :, -n:], x[:, :, :-n + end]), dim=2)
    elif dim == 3 or dim == -1:
        return torch.cat((x[:, :, :, -n:], x[:, :, :, :-n + end]), dim=3)


def mypad(x, pad, mode='constant', value=0):
    """ Function to do numpy like padding on tensors. Only works for 2-D
    padding.

    Inputs:
        x (tensor): tensor to pad
        pad (tuple): tuple of (left, right, top, bottom) pad sizes
        mode (str): 'symmetric', 'wrap', 'constant, 'reflect', 'replicate', or
            'zero'. The padding technique.
    """
    if mode == 'symmetric':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            m1, m2 = pad[2], pad[3]
            l = x.shape[-2]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, xe]
        # horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            m1, m2 = pad[0], pad[1]
            l = x.shape[-1]
            xe = reflect(np.arange(-m1, l + m2, dtype='int32'), -0.5, l - 0.5)
            return x[:, :, :, xe]
        # Both
        else:
            m1, m2 = pad[0], pad[1]
            l1 = x.shape[-1]
            xe_row = reflect(np.arange(-m1, l1 + m2, dtype='int32'), -0.5, l1 - 0.5)
            m1, m2 = pad[2], pad[3]
            l2 = x.shape[-2]
            xe_col = reflect(np.arange(-m1, l2 + m2, dtype='int32'), -0.5, l2 - 0.5)
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:, :, i, j]
    elif mode == 'periodic':
        # Vertical only
        if pad[0] == 0 and pad[1] == 0:
            xe = np.arange(x.shape[-2])
            xe = np.pad(xe, (pad[2], pad[3]), mode='wrap')
            return x[:, :, xe]
        # Horizontal only
        elif pad[2] == 0 and pad[3] == 0:
            xe = np.arange(x.shape[-1])
            xe = np.pad(xe, (pad[0], pad[1]), mode='wrap')
            return x[:, :, :, xe]
        # Both
        else:
            xe_col = np.arange(x.shape[-2])
            xe_col = np.pad(xe_col, (pad[2], pad[3]), mode='wrap')
            xe_row = np.arange(x.shape[-1])
            xe_row = np.pad(xe_row, (pad[0], pad[1]), mode='wrap')
            i = np.outer(xe_col, np.ones(xe_row.shape[0]))
            j = np.outer(np.ones(xe_col.shape[0]), xe_row)
            return x[:, :, i, j]

    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, pad, mode, value)
    elif mode == 'zero':
        return F.pad(x, pad)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def afb1d(x, h0, h1, use_amp, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    if mode == 'per' or mode == 'periodization':
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:, :, -1:]), dim=2)
            else:
                x = torch.cat((x, x[:, :, :, -1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L - 1, 0) if d == 2 else (0, L - 1)
        if use_amp:
            with torch.cuda.amp.autocast():  # for mixed precision
                lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        else:
            lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N // 2
        if d == 2:
            lohi[:, :, :L2] = lohi[:, :, :L2] + lohi[:, :, N2:N2 + L2]
            lohi = lohi[:, :, :N2]
        else:
            lohi[:, :, :, :L2] = lohi[:, :, :, :L2] + lohi[:, :, :, N2:N2 + L2]
            lohi = lohi[:, :, :, :N2]
    else:
        # Calculate the pad size
        outsize = pywt.dwt_coeff_len(N, L, mode=mode)
        p = 2 * (outsize - 1) - N + L
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p % 2 == 1:
                pad = (0, 0, 0, 1) if d == 2 else (0, 1, 0, 0)
                x = F.pad(x, pad)
            pad = (p // 2, 0) if d == 2 else (0, p // 2)
            # Calculate the high and lowpass
            if use_amp:
                with torch.cuda.amp.autocast():
                    lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
            else:
                lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (0, 0, p // 2, (p + 1) // 2) if d == 2 else (p // 2, (p + 1) // 2, 0, 0)
            x = mypad(x, pad=pad, mode=mode)
            if use_amp:
                with torch.cuda.amp.autocast():
                    lohi = F.conv2d(x, h, stride=s, groups=C)
            else:
                lohi = F.conv2d(x, h, stride=s, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return lohi


def afb1d_atrous(x, h0, h1, mode='periodic', dim=-1, dilation=1):
    """ 1D analysis filter bank (along one dimension only) of an image without
    downsampling. Does the a trous algorithm.

    Inputs:
        x (tensor): 4D input with the last two dimensions the spatial input
        h0 (tensor): 4D input for the lowpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w)
        mode (str): padding method
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
        dilation (int): dilation factor. Should be a power of 2.

    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 4
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    # Calculate the pad size
    L2 = (L * dilation) // 2
    pad = (0, 0, L2 - dilation, L2) if d == 2 else (L2 - dilation, L2, 0, 0)
    x = mypad(x, pad=pad, mode=mode)
    lohi = F.conv2d(x, h, groups=C, dilation=dilation)

    return lohi


def sfb1d(lo, hi, g0, g1, use_amp, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 4
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    N = 2 * lo.shape[d]
    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = (2, 1) if d == 2 else (1, 2)
    g0 = torch.cat([g0] * C, dim=0)
    g1 = torch.cat([g1] * C, dim=0)
    if mode == 'per' or mode == 'periodization':
        if use_amp:
            with torch.cuda.amp.autocast():
                y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
                    F.conv_transpose2d(hi, g1, stride=s, groups=C)
        else:
            y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + \
                F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:, :, :L - 2] = y[:, :, :L - 2] + y[:, :, N:N + L - 2]
            y = y[:, :, :N]
        else:
            y[:, :, :, :L - 2] = y[:, :, :, :L - 2] + y[:, :, :, N:N + L - 2]
            y = y[:, :, :, :N]
        y = roll(y, 1 - L // 2, dim=dim)
    else:
        if mode == 'zero' or mode == 'symmetric' or mode == 'reflect' or \
                mode == 'periodic':
            pad = (L - 2, 0) if d == 2 else (0, L - 2)
            if use_amp:
                with torch.cuda.amp.autocast():
                    y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                        F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
            else:
                y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
                    F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
        else:
            raise ValueError("Unkown pad type: {}".format(mode))

    return y


def mode_to_int(mode):
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode == 'per' or mode == 'periodization':
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
        return 6
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


def int_to_mode(mode):
    if mode == 0:
        return 'zero'
    elif mode == 1:
        return 'symmetric'
    elif mode == 2:
        return 'periodization'
    elif mode == 3:
        return 'constant'
    elif mode == 4:
        return 'reflect'
    elif mode == 5:
        return 'replicate'
    elif mode == 6:
        return 'periodic'
    else:
        raise ValueError("Unkown pad type: {}".format(mode))


class AFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """

    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        highs = y[:, :, 1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :, :ctx.shape[-1]]
        return dx, None, None, None, None, None


class AFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0: lowpass
        h1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        x0: Tensor of shape (N, C, L') - lowpass
        x1: Tensor of shape (N, C, L') - highpass
    """

    @staticmethod
    def forward(ctx, x, h0, h1, mode, use_amp):
        mode = int_to_mode(mode)

        # Make inputs 4d
        x = x[:, :, None, :]
        h0 = h0[:, :, None, :]
        h1 = h1[:, :, None, :]

        # Save for backwards
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[3]
        ctx.mode = mode
        ctx.use_amp = use_amp

        lohi = afb1d(x, h0, h1, use_amp, mode=mode, dim=3)
        x0 = lohi[:, ::2, 0].contiguous()
        x1 = lohi[:, 1::2, 0].contiguous()
        return x0, x1

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0, h1 = ctx.saved_tensors
            use_amp = ctx.use_amp

            # Make grads 4d
            dx0 = dx0[:, :, None, :]
            dx1 = dx1[:, :, None, :]

            dx = sfb1d(dx0, dx1, h0, h1, use_amp, mode=mode, dim=3)[:, :, 0]

            # Check for odd input
            if dx.shape[2] > ctx.shape:
                dx = dx[:, :, :ctx.shape]

        return dx, None, None, None, None, None


def afb2d(x, filts, mode='zero'):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2, 3)
            h1_col = h1
            h1_row = h1.transpose(2, 3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
    y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)

    return y


def afb2d_atrous(x, filts, mode='periodization', dilation=1):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
        dilation (int): dilation factor for the filters. Should be 2**level

    Returns:
        y: Tensor of shape (N, C, 4, H, W)
    """
    tensorize = [not isinstance(f, torch.Tensor) for f in filts]
    if len(filts) == 2:
        h0, h1 = filts
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                h0, h1, device=x.device)
        else:
            h0_col = h0
            h0_row = h0.transpose(2, 3)
            h1_col = h1
            h1_row = h1.transpose(2, 3)
    elif len(filts) == 4:
        if True in tensorize:
            h0_col, h1_col, h0_row, h1_row = prep_filt_afb2d(
                *filts, device=x.device)
        else:
            h0_col, h1_col, h0_row, h1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lohi = afb1d_atrous(x, h0_row, h1_row, mode=mode, dim=3, dilation=dilation)
    y = afb1d_atrous(lohi, h0_col, h1_col, mode=mode, dim=2, dilation=dilation)

    return y


def afb2d_nonsep(x, filts, mode='zero'):
    """ Does a 1 level 2d wavelet decomposition of an input. Doesn't do separate
    row and column filtering.

    Inputs:
        x (torch.Tensor): Input to decompose
        filts (list or torch.Tensor): If a list is given, should be the low and
            highpass filter banks. If a tensor is given, it should be of the
            form created by
            :py:func:`pytorch_wavelets.dwt.lowlevel.prep_filt_afb2d_nonsep`
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.

    Returns:
        y: Tensor of shape (N, C, 4, H, W)
    """
    C = x.shape[1]
    Ny = x.shape[2]
    Nx = x.shape[3]

    # Check the filter inputs
    if isinstance(filts, (tuple, list)):
        if len(filts) == 2:
            filts = prep_filt_afb2d_nonsep(filts[0], filts[1], device=x.device)
        else:
            filts = prep_filt_afb2d_nonsep(
                filts[0], filts[1], filts[2], filts[3], device=x.device)
    f = torch.cat([filts] * C, dim=0)
    Ly = f.shape[2]
    Lx = f.shape[3]

    if mode == 'periodization' or mode == 'per':
        if x.shape[2] % 2 == 1:
            x = torch.cat((x, x[:, :, -1:]), dim=2)
            Ny += 1
        if x.shape[3] % 2 == 1:
            x = torch.cat((x, x[:, :, :, -1:]), dim=3)
            Nx += 1
        pad = (Ly - 1, Lx - 1)
        stride = (2, 2)
        x = roll(roll(x, -Ly // 2, dim=2), -Lx // 2, dim=3)
        y = F.conv2d(x, f, padding=pad, stride=stride, groups=C)
        y[:, :, :Ly // 2] += y[:, :, Ny // 2:Ny // 2 + Ly // 2]
        y[:, :, :, :Lx // 2] += y[:, :, :, Nx // 2:Nx // 2 + Lx // 2]
        y = y[:, :, :Ny // 2, :Nx // 2]
    elif mode == 'zero' or mode == 'symmetric' or mode == 'reflect':
        # Calculate the pad size
        out1 = pywt.dwt_coeff_len(Ny, Ly, mode=mode)
        out2 = pywt.dwt_coeff_len(Nx, Lx, mode=mode)
        p1 = 2 * (out1 - 1) - Ny + Ly
        p2 = 2 * (out2 - 1) - Nx + Lx
        if mode == 'zero':
            # Sadly, pytorch only allows for same padding before and after, if
            # we need to do more padding after for odd length signals, have to
            # prepad
            if p1 % 2 == 1 and p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 1))
            elif p1 % 2 == 1:
                x = F.pad(x, (0, 0, 0, 1))
            elif p2 % 2 == 1:
                x = F.pad(x, (0, 1, 0, 0))
            # Calculate the high and lowpass
            y = F.conv2d(
                x, f, padding=(p1 // 2, p2 // 2), stride=2, groups=C)
        elif mode == 'symmetric' or mode == 'reflect' or mode == 'periodic':
            pad = (p2 // 2, (p2 + 1) // 2, p1 // 2, (p1 + 1) // 2)
            x = mypad(x, pad=pad, mode=mode)
            y = F.conv2d(x, f, stride=2, groups=C)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return y


def sfb2d(ll, lh, hl, hh, filts, mode='zero'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does separate row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.sfb1d`

    Inputs:
        ll (torch.Tensor): lowpass coefficients
        lh (torch.Tensor): horizontal coefficients
        hl (torch.Tensor): vertical coefficients
        hh (torch.Tensor): diagonal coefficients
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    tensorize = [not isinstance(x, torch.Tensor) for x in filts]
    if len(filts) == 2:
        g0, g1 = filts
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(g0, g1)
        else:
            g0_col = g0
            g0_row = g0.transpose(2, 3)
            g1_col = g1
            g1_row = g1.transpose(2, 3)
    elif len(filts) == 4:
        if True in tensorize:
            g0_col, g1_col, g0_row, g1_row = prep_filt_sfb2d(*filts)
        else:
            g0_col, g1_col, g0_row, g1_row = filts
    else:
        raise ValueError("Unknown form for input filts")

    lo = sfb1d(ll, lh, g0_col, g1_col, mode=mode, dim=2)
    hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
    y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)

    return y


class SFB2D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """

    @staticmethod
    def forward(ctx, low, highs, g0_row, g1_row, g0_col, g1_col, mode):
        mode = int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_row, g1_row, g0_col, g1_col)

        lh, hl, hh = torch.unbind(highs, dim=2)
        lo = sfb1d(low, lh, g0_col, g1_col, mode=mode, dim=2)
        hi = sfb1d(hl, hh, g0_col, g1_col, mode=mode, dim=2)
        y = sfb1d(lo, hi, g0_row, g1_row, mode=mode, dim=3)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_row, g1_row, g0_col, g1_col = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=3)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 4, s[-2], s[-1])
            dlow = dx[:, :, 0].contiguous()
            dhigh = dx[:, :, 1:].contiguous()
        return dlow, dhigh, None, None, None, None, None


class SFB1D(Function):
    """ Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        low (torch.Tensor): Lowpass to reconstruct of shape (N, C, L)
        high (torch.Tensor): Highpass to reconstruct of shape (N, C, L)
        g0: lowpass
        g1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        y: Tensor of shape (N, C*2, L')
    """

    @staticmethod
    def forward(ctx, low, high, g0, g1, mode, use_amp):
        mode = int_to_mode(mode)
        # Make into a 2d tensor with 1 row
        low = low[:, :, None, :]
        high = high[:, :, None, :]
        g0 = g0[:, :, None, :]
        g1 = g1[:, :, None, :]

        ctx.mode = mode
        ctx.save_for_backward(g0, g1)
        ctx.use_amp = use_amp

        return sfb1d(low, high, g0, g1, use_amp, mode=mode, dim=3)[:, :, 0]

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            use_amp = ctx.use_amp
            g0, g1, = ctx.saved_tensors
            dy = dy[:, :, None, :]

            dx = afb1d(dy, g0, g1, use_amp, mode=mode, dim=3)

            dlow = dx[:, ::2, 0].contiguous()
            dhigh = dx[:, 1::2, 0].contiguous()
        return dlow, dhigh, None, None, None, None, None


def sfb2d_nonsep(coeffs, filts, mode='zero'):
    """ Does a single level 2d wavelet reconstruction of wavelet coefficients.
    Does not do separable filtering.

    Inputs:
        coeffs (torch.Tensor): tensor of coefficients of shape (N, C, 4, H, W)
            where the third dimension indexes across the (ll, lh, hl, hh) bands.
        filts (list of ndarray or torch.Tensor): If a list of tensors has been
            given, this function assumes they are in the right form (the form
            returned by
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d_nonsep`).
            Otherwise, this function will prepare the filters to be of the right
            form by calling
            :py:func:`~pytorch_wavelets.dwt.lowlevel.prep_filt_sfb2d_nonsep`.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. Which
            padding to use. If periodization, the output size will be half the
            input size.  Otherwise, the output size will be slightly larger than
            half.
    """
    C = coeffs.shape[1]
    Ny = coeffs.shape[-2]
    Nx = coeffs.shape[-1]

    # Check the filter inputs - should be in the form of a torch tensor, but if
    # not, tensorize it here.
    if isinstance(filts, (tuple, list)):
        if len(filts) == 2:
            filts = prep_filt_sfb2d_nonsep(filts[0], filts[1],
                                           device=coeffs.device)
        elif len(filts) == 4:
            filts = prep_filt_sfb2d_nonsep(
                filts[0], filts[1], filts[2], filts[3], device=coeffs.device)
        else:
            raise ValueError("Unkown form for input filts")
    f = torch.cat([filts] * C, dim=0)
    Ly = f.shape[2]
    Lx = f.shape[3]

    x = coeffs.reshape(coeffs.shape[0], -1, coeffs.shape[-2], coeffs.shape[-1])
    if mode == 'periodization' or mode == 'per':
        ll = F.conv_transpose2d(x, f, groups=C, stride=2)
        ll[:, :, :Ly - 2] += ll[:, :, 2 * Ny:2 * Ny + Ly - 2]
        ll[:, :, :, :Lx - 2] += ll[:, :, :, 2 * Nx:2 * Nx + Lx - 2]
        ll = ll[:, :, :2 * Ny, :2 * Nx]
        ll = roll(roll(ll, 1 - Ly // 2, dim=2), 1 - Lx // 2, dim=3)
    elif mode == 'symmetric' or mode == 'zero' or mode == 'reflect' or \
            mode == 'periodic':
        pad = (Ly - 2, Lx - 2)
        ll = F.conv_transpose2d(x, f, padding=pad, groups=C, stride=2)
    else:
        raise ValueError("Unkown pad type: {}".format(mode))

    return ll.contiguous()


def prep_filt_afb2d_nonsep(h0_col, h1_col, h0_row=None, h1_row=None,
                           device=None):
    """
    Prepares the filters to be of the right form for the afb2d_nonsep function.
    In particular, makes 2d point spread functions, and mirror images them in
    preparation to do torch.conv2d.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        filts: (4, 1, h, w) tensor ready to get the four subbands
    """
    h0_col = np.array(h0_col).ravel()
    h1_col = np.array(h1_col).ravel()
    if h0_row is None:
        h0_row = h0_col
    if h1_row is None:
        h1_row = h1_col
    ll = np.outer(h0_col, h0_row)
    lh = np.outer(h1_col, h0_row)
    hl = np.outer(h0_col, h1_row)
    hh = np.outer(h1_col, h1_row)
    filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                      hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]], axis=0)
    filts = torch.tensor(filts, dtype=torch.get_default_dtype(), device=device)
    return filts


def prep_filt_sfb2d_nonsep(g0_col, g1_col, g0_row=None, g1_row=None,
                           device=None):
    """
    Prepares the filters to be of the right form for the sfb2d_nonsep function.
    In particular, makes 2d point spread functions. Does not mirror image them
    as sfb2d_nonsep uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        filts: (4, 1, h, w) tensor ready to combine the four subbands
    """
    g0_col = np.array(g0_col).ravel()
    g1_col = np.array(g1_col).ravel()
    if g0_row is None:
        g0_row = g0_col
    if g1_row is None:
        g1_row = g1_col
    ll = np.outer(g0_col, g0_row)
    lh = np.outer(g1_col, g0_row)
    hl = np.outer(g0_col, g1_row)
    hh = np.outer(g1_col, g1_row)
    filts = np.stack([ll[None], lh[None], hl[None], hh[None]], axis=0)
    filts = torch.tensor(filts, dtype=torch.get_default_dtype(), device=device)
    return filts


def prep_filt_sfb2d(g0_col, g1_col, g0_row=None, g1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the sfb2d function.  In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0_col (array-like): low pass column filter bank
        g1_col (array-like): high pass column filter bank
        g0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        g1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (g0_col, g1_col, g0_row, g1_row)
    """
    g0_col, g1_col = prep_filt_sfb1d(g0_col, g1_col, device)
    if g0_row is None:
        g0_row, g1_row = g0_col, g1_col
    else:
        g0_row, g1_row = prep_filt_sfb1d(g0_row, g1_row, device)

    g0_col = g0_col.reshape((1, 1, -1, 1))
    g1_col = g1_col.reshape((1, 1, -1, 1))
    g0_row = g0_row.reshape((1, 1, 1, -1))
    g1_row = g1_row.reshape((1, 1, 1, -1))

    return g0_col, g1_col, g0_row, g1_row


def prep_filt_sfb1d(g0, g1, device=None):
    """
    Prepares the filters to be of the right form for the sfb1d function. In
    particular, makes the tensors the right shape. It does not mirror image them
    as as sfb2d uses conv2d_transpose which acts like normal convolution.

    Inputs:
        g0 (array-like): low pass filter bank
        g1 (array-like): high pass filter bank
        device: which device to put the tensors on to

    Returns:
        (g0, g1)
    """
    g0 = np.array(g0).ravel()
    g1 = np.array(g1).ravel()
    t = torch.get_default_dtype()
    g0 = torch.tensor(g0, device=device, dtype=t).reshape((1, 1, -1))
    g1 = torch.tensor(g1, device=device, dtype=t).reshape((1, 1, -1))

    return g0, g1


def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0_col (array-like): low pass column filter bank
        h1_col (array-like): high pass column filter bank
        h0_row (array-like): low pass row filter bank. If none, will assume the
            same as column filter
        h1_row (array-like): high pass row filter bank. If none, will assume the
            same as column filter
        device: which device to put the tensors on to

    Returns:
        (h0_col, h1_col, h0_row, h1_row)
    """
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_row = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)

    h0_col = h0_col.reshape((1, 1, -1, 1))
    h1_col = h1_col.reshape((1, 1, -1, 1))
    h0_row = h0_row.reshape((1, 1, 1, -1))
    h1_row = h1_row.reshape((1, 1, 1, -1))
    return h0_col, h1_col, h0_row, h1_row


def prep_filt_afb1d(h0, h1, device=None):
    """
    Prepares the filters to be of the right form for the afb2d function.  In
    particular, makes the tensors the right shape. It takes mirror images of
    them as as afb2d uses conv2d which acts like normal correlation.

    Inputs:
        h0 (array-like): low pass column filter bank
        h1 (array-like): high pass column filter bank
        device: which device to put the tensors on to

    Returns:
        (h0, h1)
    """
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1, 1, -1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1, 1, -1))
    return h0, h1


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)