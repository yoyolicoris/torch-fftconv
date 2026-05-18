import torch
import torch.nn.functional as F
from torch_fftconv.functional import (
    fft_conv1d,
    fft_conv2d,
    fft_conv3d,
    fft_conv_transpose1d,
    fft_conv_transpose2d,
    fft_conv_transpose3d,
)
import pytest
from itertools import product

torch.set_float32_matmul_precision("highest")
torch.backends.fp32_precision = "ieee"

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("in_channels", [4, 12])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("length", [1027])
@pytest.mark.parametrize("kernel_size", [128, 256])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("groups", [1, 4])
def test_cmplx_conv1d(
    batch,
    length,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
):

    x = torch.randn(
        batch,
        in_channels,
        length,
        requires_grad=True,
        device=device,
        dtype=torch.complex64,
    )
    weight = (
        torch.randn(
            out_channels,
            in_channels // groups,
            kernel_size,
            device=device,
            dtype=torch.complex64,
        )
        / (kernel_size * in_channels // groups) ** 0.5
    )
    if bias:
        bias = torch.randn(out_channels, device=device, dtype=torch.complex64)
    else:
        bias = None

    y1 = F.conv1d(x, weight, bias, stride, padding, dilation, groups)
    y2 = fft_conv1d(x, weight, bias, stride, padding, dilation, groups)
    assert torch.allclose(y1, y2, atol=6e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.abs().sum().backward()


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("in_channels", [8, 32])
@pytest.mark.parametrize("out_channels", [4, 16])
@pytest.mark.parametrize("length", [(101, 101)])
@pytest.mark.parametrize("kernel_size", [17, 23])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("groups", [1, 2])
def test_cmplx_conv2d(
    batch,
    length,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
):

    x = torch.randn(
        batch,
        in_channels,
        *length,
        requires_grad=True,
        device=device,
        dtype=torch.complex64,
    )
    weight = (
        torch.randn(
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
            device=device,
            dtype=torch.complex64,
        )
        / (kernel_size * kernel_size * in_channels // groups) ** 0.5
    )
    if bias:
        bias = torch.randn(out_channels, device=device, dtype=torch.complex64)
    else:
        bias = None

    y1 = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    y2 = fft_conv2d(x, weight, bias, stride, padding, dilation, groups)
    assert torch.allclose(y1, y2, atol=6e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.abs().sum().backward()


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("in_channels", [8])
@pytest.mark.parametrize("out_channels", [8])
@pytest.mark.parametrize("length", [(53, 53, 59)])
@pytest.mark.parametrize("kernel_size", [9, 11])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("padding", [6])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("groups", [1, 2])
def test_cmplx_conv3d(
    batch,
    length,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
):

    x = torch.randn(
        batch,
        in_channels,
        *length,
        requires_grad=True,
        device=device,
        dtype=torch.complex64,
    )
    weight = (
        torch.randn(
            out_channels,
            in_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size,
            device=device,
            dtype=torch.complex64,
        )
        / (kernel_size * kernel_size * kernel_size * in_channels // groups) ** 0.5
    )
    if bias:
        bias = torch.randn(out_channels, device=device, dtype=torch.complex64)
    else:
        bias = None

    y1 = F.conv3d(x, weight, bias, stride, padding, dilation, groups)
    y2 = fft_conv3d(x, weight, bias, stride, padding, dilation, groups)
    assert torch.allclose(y1, y2, atol=6e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.abs().sum().backward()


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("in_channels", [4, 12])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("length", [409])
@pytest.mark.parametrize("kernel_size", [128, 256])
@pytest.mark.parametrize(
    "stride,dilation,output_padding",
    [(1, 1, 0), (3, 2, 0)],
)
@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("groups", [1, 4])
def test_cmplx_conv_transpose1d(
    batch,
    length,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    bias,
):

    x = torch.randn(
        batch,
        in_channels,
        length,
        requires_grad=True,
        device=device,
        dtype=torch.complex64,
    )
    weight = (
        torch.randn(
            in_channels,
            out_channels // groups,
            kernel_size,
            device=device,
            dtype=torch.complex64,
        )
        / (kernel_size * kernel_size * in_channels // groups) ** 0.5
    )
    if bias:
        bias = torch.randn(out_channels, device=device, dtype=torch.complex64)
    else:
        bias = None

    y1 = F.conv_transpose1d(
        x, weight, bias, stride, padding, output_padding, groups, dilation
    )
    y2 = fft_conv_transpose1d(
        x, weight, bias, stride, padding, output_padding, groups, dilation
    )
    assert torch.allclose(y1, y2, atol=6e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.abs().sum().backward()


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("in_channels", [8, 32])
@pytest.mark.parametrize("out_channels", [4, 16])
@pytest.mark.parametrize("length", [(31, 31)])
@pytest.mark.parametrize("kernel_size", [17, 23])
@pytest.mark.parametrize("padding", [0, 7])
@pytest.mark.parametrize(
    "stride,dilation,output_padding",
    [(1, 1, 0), (3, 2, 0)],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("groups", [1, 2])
def test_cmplx_conv_transpose2d(
    batch,
    length,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    bias,
):

    x = torch.randn(
        batch,
        in_channels,
        *length,
        requires_grad=True,
        device=device,
        dtype=torch.complex64,
    )
    weight = (
        torch.randn(
            in_channels,
            out_channels // groups,
            kernel_size,
            kernel_size,
            device=device,
            dtype=torch.complex64,
        )
        / (kernel_size * kernel_size * in_channels // groups) ** 0.5
    )
    if bias:
        bias = torch.randn(out_channels, device=device, dtype=torch.complex64)
    else:
        bias = None

    y1 = F.conv_transpose2d(
        x,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups=groups,
        dilation=dilation,
    )
    y2 = fft_conv_transpose2d(
        x,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups=groups,
        dilation=dilation,
    )
    assert torch.allclose(y1, y2, atol=6e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.abs().sum().backward()


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("in_channels", [4])
@pytest.mark.parametrize("out_channels", [6])
@pytest.mark.parametrize("length", [(29, 23, 23)])
@pytest.mark.parametrize("kernel_size", [9, 11])
@pytest.mark.parametrize("padding", [6])
@pytest.mark.parametrize(
    "stride,dilation,output_padding",
    [(1, 1, 0), (3, 2, 0)],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("groups", [1, 2])
def test_cmplx_conv_transpose3d(
    batch,
    length,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    bias,
):

    x = torch.randn(
        batch,
        in_channels,
        *length,
        requires_grad=True,
        device=device,
        dtype=torch.complex64,
    )
    weight = (
        torch.randn(
            in_channels,
            out_channels // groups,
            kernel_size,
            kernel_size,
            kernel_size,
            device=device,
            dtype=torch.complex64,
        )
        / (kernel_size * kernel_size * in_channels // groups) ** 0.5
    )
    if bias:
        bias = torch.randn(out_channels, device=device, dtype=torch.complex64)
    else:
        bias = None

    y1 = F.conv_transpose3d(
        x,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups=groups,
        dilation=dilation,
    )
    y2 = fft_conv_transpose3d(
        x,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups=groups,
        dilation=dilation,
    )
    assert torch.allclose(y1, y2, atol=6e-5, rtol=1e-5), torch.abs(y1 - y2).max().item()
    y2.abs().sum().backward()
