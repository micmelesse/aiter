# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import contextlib
from functools import partial
from unittest.mock import Mock

import pytest
import torch

from aiter.ops.triton.attention import mha
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_impl,
)
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_with_tol,
    generate_qkv,
    generate_random_padding_mask,
)


@contextlib.contextmanager
def _with_default_device(device):
    """Run the capture under an explicit torch default device, restored on exit."""
    prev_device = torch.get_default_device()
    try:
        torch.set_default_device(device)
        yield
    finally:
        torch.set_default_device(prev_device)


@pytest.fixture
def dao_ai_impl():
    """Set mha impl to dao_ai for the test, restore to default on cleanup."""
    mha_set_impl("dao_ai")
    yield
    mha_set_impl("default")


# fmt: off
CORRECTNESS_CASES = [
    (1, 128, 128, 8, 8, 64, True, False, False, (-1, -1), 0.0),  # fwd causal
    (2, 256, 256, 16, 4, 128, False, False, False, (-1, -1), 0.0),  # fwd GQA non-causal
    (1, 128, 128, 8, 8, 64, True, True, False, (-1, -1), 0.0),  # fwd_varlen causal
    (1, 128, 128, 8, 8, 64, True, False, True, (-1, -1), 0.0),  # bwd causal
    (1, 128, 128, 8, 8, 64, True, True, True, (-1, -1), 0.0),  # bwd_varlen causal
    (1, 128, 128, 8, 8, 64, True, False, True, (32, 0), 0.0),  # causal window
    (1, 128, 128, 8, 8, 64, True, True, True, (32, 0), 0.0),  # causal window varlen
    (1, 128, 128, 8, 8, 64, False, False, True, (16, 16), 0.0),  # symmetric window
    (1, 128, 128, 8, 8, 64, False, True, True, (16, 16), 0.0),  # symmetric win varlen
    (1, 128, 128, 8, 8, 64, False, False, True, (-1, 32), 0.0),  # infinite-left window
    # Infinite-RIGHT window (L, -1): the mirror of the infinite-left row above.
    # MUST be non-causal -- under causal the right edge is capped to the diagonal
    # (attention_ref forces window=(L, 0)), so (L, -1) would collapse to (L, 0)
    # and never exercise the new unbounded-right path.
    (1, 128, 128, 8, 8, 64, False, False, True, (32, -1), 0.0),  # infinite-right window
    (1, 128, 128, 8, 8, 64, False, True, True, (32, -1), 0.0),  # infinite-right varlen
    # seqlen_q != seqlen_k exercises the causal_offset / delta_qk arithmetic
    (1, 128, 256, 8, 8, 64, True, False, True, (32, 0), 0.0),  # causal, sq != sk
    (1, 128, 256, 8, 8, 64, True, True, True, (32, 0), 0.0),  # causal varlen, sq != sk
    (1, 128, 256, 8, 8, 64, False, False, True, (16, 16), 0.0),  # symmetric, sq != sk
    # infinite-right with sq != sk: exercises the sk-sq offset in the
    # unbounded-right per-element mask and the _sliding_window bounds.
    (1, 128, 256, 8, 8, 64, False, False, True, (32, -1), 0.0),  # infinite-right sq != sk
    # GQA (num_q_heads != num_k_heads) combined with sliding-window backward
    (2, 128, 128, 16, 4, 64, True, False, True, (32, 0), 0.0),  # GQA causal window
    (2, 128, 128, 16, 4, 64, False, True, True, (16, 16), 0.0),  # GQA symmetric varlen
    # Production-size sequence lengths (beyond the upstream 2048 cap) with
    # large windows (128/256). seqlen >> window means the window spans many
    # key blocks, exercising the bwd full/partial/skipped block
    # classification that the small (seqlen 128, window 16/32) cases barely
    # reach.
    (1, 4096, 4096, 8, 8, 64, True, False, True, (256, 0), 0.0),  # large causal window
    (1, 4096, 4096, 8, 8, 64, True, True, True, (256, 0), 0.0),  # large causal varlen
    (1, 4096, 4096, 8, 8, 64, False, False, True, (256, 256), 0.0),  # large symmetric
    (2, 4096, 4096, 16, 4, 64, True, False, True, (128, 0), 0.0),  # GQA large causal
    # seqlen_q != seqlen_k at production size exercises the causal_offset /
    # delta_qk arithmetic with a window spanning many blocks.
    (1, 4096, 8192, 8, 8, 64, True, False, True, (256, 0), 0.0),  # large causal sq != sk
    # Padded last block: seqlen NOT a multiple of the block size, with an active
    # window -> exercises handle_padded_last_block (the last partial-K-block bucket
    # move). Every multiple-of-block case above skips this branch. sq == sk keeps
    # every query row with at least the diagonal key (no fully-masked rows).
    (1, 1023, 1023, 8, 8, 64, True, False, True, (128, 0), 0.0),  # causal
    (1, 1025, 1025, 8, 8, 64, False, False, True, (64, 64), 0.0),  # symmetric
    (1, 4097, 4097, 8, 8, 64, True, False, True, (256, 0), 0.0),  # production size
    (1, 1024, 1024, 8, 8, 64, False, False, True, (0, 0), 0.0),  # (0,0) band
    # Broaden the production block-skip regime past hd64 / MHA: block sizes are
    # head-dim dependent and MQA changes the block-classification arithmetic.
    (1, 4096, 4096, 8, 8, 128, True, False, True, (256, 0), 0.0),  # hd128
    (2, 4096, 4096, 8, 1, 64, True, False, True, (256, 0), 0.0),  # MQA
    # Dropout: output and gradients against a reference that applies the record the
    # forward returned. Random varlen padding also checks the record's padding.
    (2, 96, 96, 4, 4, 64, False, False, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, True, False, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, False, False, True, (16, 16), 0.17),
    (2, 96, 96, 4, 4, 64, False, True, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, True, True, True, (-1, -1), 0.17),
    (2, 96, 96, 4, 4, 64, False, True, True, (16, 16), 0.17),
]
# fmt: on


@pytest.mark.parametrize(
    "BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, CAUSAL, VARLEN, BWD, WINDOW_SIZE, DROPOUT",
    CORRECTNESS_CASES,
)
def test_mha_dao_ai_correctness(
    dao_ai_impl,
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    VARLEN: bool,
    BWD: bool,
    WINDOW_SIZE: tuple[int, int],
    DROPOUT: float,
    dtype=torch.bfloat16,
):
    """dao_ai forward and backward, dense and varlen, match the PyTorch reference. With
    dropout, the reference applies the dropout record that the forward returned."""
    dense = partial(flash_attn_func, implementation="dao_ai")
    varlen = partial(flash_attn_varlen_func, implementation="dao_ai")
    # Per-call selection must win over the process default, including in backward.
    mha_set_impl("default")
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q = torch.randn(
        BATCH,
        SEQLEN_Q,
        NUM_Q_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )
    k = torch.randn(
        BATCH,
        SEQLEN_K,
        NUM_K_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )
    v = torch.randn(
        BATCH,
        SEQLEN_K,
        NUM_K_HEADS,
        HEAD_SZ,
        device="cuda",
        dtype=dtype,
        requires_grad=BWD,
    )

    if VARLEN:
        query_padding_mask = generate_random_padding_mask(
            SEQLEN_Q, BATCH, "cuda", mode="random" if DROPOUT else "full"
        )
        key_padding_mask = generate_random_padding_mask(
            SEQLEN_K, BATCH, "cuda", mode="random" if DROPOUT else "full"
        )
        (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        q_unpad.requires_grad_(BWD)
        k_unpad.requires_grad_(BWD)
        v_unpad.requires_grad_(BWD)

        with torch.set_grad_enabled(BWD):
            triton_out = varlen(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                dropout_p=DROPOUT,
                return_lse=True,
                return_attn_probs=True,
            )
    else:
        query_padding_mask = None
        key_padding_mask = None
        with torch.set_grad_enabled(BWD):
            triton_out = dense(
                q,
                k,
                v,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                dropout_p=DROPOUT,
                return_lse=True,
                return_attn_probs=True,
            )

    triton_out, _, scores = triton_out
    mask = None
    if DROPOUT:
        # The varlen record covers the longest real sequence; the rest is dropped.
        scores = scores[:, :, :SEQLEN_Q, :SEQLEN_K]
        pad = (0, SEQLEN_K - scores.shape[3], 0, SEQLEN_Q - scores.shape[2])
        mask = torch.nn.functional.pad(scores, pad, value=-1.0) >= 0

    # Reference + tolerances. For backward, attention_ref_with_tol derives each
    # tensor's atol from the fp32-vs-bf16 reference gap (upstream FA pattern, as in
    # test_mha_v3): bf16 gradient reductions -- largest under MQA / long seqlen --
    # outrun a fixed 1e-2 atol.
    do = torch.randn_like(q) if BWD else None
    if BWD:
        torch_out, (torch_dq, torch_dk, torch_dv), fwd_tol, bwd_tols = (
            attention_ref_with_tol(
                q,
                k,
                v,
                do,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                dropout_p=DROPOUT,
                dropout_mask=mask,
            )
        )
    else:
        torch_out, _, _ = attention_ref(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=WINDOW_SIZE,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=DROPOUT,
            dropout_mask=mask,
        )
        fwd_tol = (1e-2, 1e-2)

    # Forward check against PyTorch reference
    fwd_atol, fwd_rtol = fwd_tol
    triton_out_fwd = output_pad_fn(triton_out) if VARLEN else triton_out
    torch.testing.assert_close(triton_out_fwd, torch_out, atol=fwd_atol, rtol=fwd_rtol)

    # Backward check against PyTorch reference
    if BWD:
        if VARLEN:
            triton_out = output_pad_fn(triton_out)
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out, (q_unpad, k_unpad, v_unpad), do
            )
            triton_dq = dq_pad_fn(triton_dq)
            triton_dk = dk_pad_fn(triton_dk)
            triton_dv = dk_pad_fn(triton_dv)
        else:
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out, (q, k, v), do
            )

        for tri, ref, (atol, rtol), name in zip(
            (triton_dq, triton_dk, triton_dv),
            (torch_dq, torch_dk, torch_dv),
            bwd_tols,
            ("dq", "dk", "dv"),
        ):
            torch.testing.assert_close(
                tri,
                ref,
                atol=atol,
                rtol=rtol,
                msg=lambda m, name=name: f"dao_ai bwd {name} mismatch\n\n{m}\n",
            )

        if DROPOUT:
            # The gradients are closer to the forward's dropout mask than to a shifted
            # one, as they would not be if backward lost the forward's RNG state.
            _, wrong_grads, _, _ = attention_ref_with_tol(
                q,
                k,
                v,
                do,
                causal=CAUSAL,
                window_size=WINDOW_SIZE,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                dropout_p=DROPOUT,
                dropout_mask=torch.roll(mask, 1, dims=-1),
            )
            for tri, ref, wrong, name in zip(
                (triton_dq, triton_dk, triton_dv),
                (torch_dq, torch_dk, torch_dv),
                wrong_grads,
                ("dq", "dk", "dv"),
            ):
                assert (tri - ref).abs().max() < (tri - wrong).abs().max(), name


@pytest.mark.parametrize("return_softmax", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
def test_mha_dao_ai_contract(varlen, return_softmax):
    """_flash_attn_forward returns rng_state on q's device and the dropout record in q's
    dtype, empty when not requested, as FlashAttention's fake ops declare."""
    q = torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.float16)
    seqlens = {}
    if varlen:
        q = q.flatten(0, 1)
        cu_seqlens = torch.arange(0, 3 * 128, 128, device="cuda", dtype=torch.int32)
        seqlens = {"cu_seqlens_q": cu_seqlens, "cu_seqlens_k": cu_seqlens}
    _, _, s_dmask, _, _, rng_state = mha._flash_attn_forward(
        q,
        q,
        q,
        0.17,
        64**-0.5,
        True,
        -1,
        -1,
        None,
        None,
        True,
        return_softmax,
        128,
        128,
        **seqlens,
        implementation="dao_ai",
    )
    assert rng_state.device == q.device
    assert rng_state.shape == (2,) and rng_state.dtype == torch.int64
    assert s_dmask.dtype == q.dtype
    assert s_dmask.shape == ((2, 4, 128, 128) if return_softmax else (0,))


@pytest.mark.parametrize("head_size", [53, 64])
@pytest.mark.parametrize("varlen", [False, True])
def test_mha_dao_ai_routing(monkeypatch, varlen, head_size):
    """Each forward's backward runs that forward's implementation, even after
    mha_set_impl changes the default between them."""
    dao_bwd = Mock(wraps=getattr(mha.flash_attn_2, "varlen_bwd" if varlen else "bwd"))
    default_bwd = Mock(wraps=mha.flash_attn_onekernel_backward)
    monkeypatch.setattr(mha.flash_attn_2, "varlen_bwd" if varlen else "bwd", dao_bwd)
    monkeypatch.setattr(mha, "flash_attn_onekernel_backward", default_bwd)
    shape = (16, 4, head_size) if varlen else (2, 8, 4, head_size)
    q, k, v = (
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    cu_seqlens = torch.tensor([0, 8, 16], device="cuda", dtype=torch.int32)

    def attention(**kwargs):
        if varlen:
            return flash_attn_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, 8, 8, **kwargs
            )
        return flash_attn_func(q, k, v, **kwargs)

    try:
        mha_set_impl("dao_ai")
        dao_out = attention(deterministic=True)
        default_out = attention(implementation="default")

        mha_set_impl("default")
        torch.autograd.grad(dao_out, (q, k, v), torch.ones_like(dao_out))
        assert dao_bwd.call_count == 1 and dao_bwd.call_args.kwargs["deterministic"]
        assert default_bwd.call_count == 0

        mha_set_impl("dao_ai")
        grads = torch.autograd.grad(
            default_out, (q, k, v), torch.ones_like(default_out)
        )
        assert default_bwd.call_count == 1 and dao_bwd.call_count == 1
        assert [g.shape for g in grads] == [q.shape, k.shape, v.shape]
    finally:
        mha_set_impl("default")


@pytest.mark.parametrize(
    "default, call, kwargs, error, message",
    [
        ("dao_ai", "mxfp8", {}, ValueError, "not supported by the dao_ai"),
        (
            "default",
            "mxfp8",
            {"implementation": "dao_ai"},
            ValueError,
            "not supported by the dao_ai",
        ),
        (
            "default",
            "mxfp8",
            {"implementation": "unknown"},
            ValueError,
            "Unknown MHA implementation",
        ),
        ("dao_ai", "block_scale", {}, ValueError, "not supported by the dao_ai"),
        (
            "default",
            "block_scale",
            {"implementation": "dao_ai"},
            ValueError,
            "not supported by the dao_ai",
        ),
        (
            "default",
            "block_scale",
            {"implementation": "unknown"},
            ValueError,
            "Unknown MHA implementation",
        ),
        (
            "default",
            "dense",
            {"implementation": "unknown"},
            ValueError,
            "Unknown MHA implementation",
        ),
        (
            "default",
            "varlen",
            {"implementation": "unknown"},
            ValueError,
            "Unknown MHA implementation",
        ),
        (
            "default",
            "varlen",
            {"implementation": "dao_ai", "block_table": True},
            NotImplementedError,
            "",
        ),
        (
            "default",
            "varlen",
            {"implementation": "dao_ai", "out": True},
            NotImplementedError,
            "",
        ),
        ("default", "set_impl", {}, ValueError, "Unknown MHA implementation"),
    ],
)
def test_mha_dao_ai_refusal(monkeypatch, default, call, kwargs, error, message):
    """Unsupported uses of the implementation selector fail with a named error before
    any forward runs, and leave the default implementation unchanged."""
    forward = Mock(wraps=mha._flash_attn_forward)
    monkeypatch.setattr(mha, "_flash_attn_forward", forward)
    q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    buffer = torch.zeros(1, device="cuda", dtype=torch.int32)
    kwargs = {
        name: buffer if value is True else value for name, value in kwargs.items()
    }
    mha_set_impl(default)
    try:
        with pytest.raises(error, match=message):
            if call == "mxfp8":
                flash_attn_func(
                    q,
                    q,
                    q,
                    q_mx_scale=scale,
                    k_mx_scale=scale,
                    v_mx_scale=scale,
                    **kwargs,
                )
            elif call == "block_scale":
                descale = torch.ones(1, 4, 2, device="cuda", dtype=torch.float32)
                flash_attn_func(
                    q,
                    q,
                    q,
                    q_block_descale=descale,
                    k_block_descale=descale,
                    v_scale=scale,
                    **kwargs,
                )
            elif call == "dense":
                flash_attn_func(q, q, q, **kwargs)
            elif call == "varlen":
                cu_seqlens = torch.tensor(
                    [0, 64, 128], device="cuda", dtype=torch.int32
                )
                flash_attn_varlen_func(
                    q[0], q[0], q[0], cu_seqlens, cu_seqlens, 64, 64, **kwargs
                )
            else:
                mha_set_impl("unknown")
        forward.assert_not_called()
        assert mha._MHA_IMPL == default
    finally:
        mha_set_impl("default")


@pytest.mark.parametrize("train", [False, True], ids=["forward", "dropout-train-step"])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
def test_mha_dao_ai_capture(dao_ai_impl, mha_type, default_device, varlen, train):
    """A forward, or a dropout training step, captured in a CUDA graph under either
    default device replays to the eager result: nothing allocates on the default device
    or syncs the host."""
    torch.manual_seed(20)
    batch, seqlen, nheads, d = 2, 128, 8, 128
    nheads_k = nheads if mha_type == "mha" else 2
    q = torch.randn(batch, seqlen, nheads, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads_k, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads_k, d, device="cuda", dtype=torch.bfloat16)
    options = {"causal": True, "dropout_p": 0.17 if train else 0.0}
    if varlen:
        query_padding_mask = generate_random_padding_mask(
            seqlen, batch, "cuda", mode="full"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen, batch, "cuda", mode="full"
        )
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *_ = (
            generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
        )
        lengths = (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        attention = partial(flash_attn_varlen_func, **options)
    else:
        lengths = ()
        attention = partial(flash_attn_func, **options)
    inputs = tuple(x.detach().requires_grad_(train) for x in (q, k, v))
    dout = torch.randn_like(inputs[0])

    def step():
        out = attention(*inputs, *lengths)
        return torch.autograd.grad(out, inputs, dout) if train else (out,)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # warmup (Triton JIT)
            step()
    torch.cuda.current_stream().wait_stream(stream)

    g = torch.cuda.CUDAGraph()
    with _with_default_device(default_device), torch.cuda.graph(g):
        graph = step()
    g.replay()
    torch.cuda.synchronize()

    for replayed, eager in zip(graph, step()):
        assert not torch.isnan(replayed).any(), "Graph replay produced NaN"
        if train:
            torch.testing.assert_close(replayed, eager)
        else:
            torch.testing.assert_close(replayed, eager, atol=1e-5, rtol=0)
