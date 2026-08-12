"""
Unit tests for transformer_MRI modules.
Author: Zhibo Zhu. Date: 07/23/2026.
"""
import torch
import torch.nn as nn
import numpy as np
import sys

# Import all modules
from embedding import embedding
from positional_encoding import positional_encoding
from attention_block import multi_head_self_attn, multi_head_cross_attn
from encoder_block import encoder_block
from decoder_block import decoder_block
from refine_module import refine_module
from transformer import transformer


def test_embedding():
    """Test embedding layer."""
    print("\n" + "="*50)
    print("Testing Embedding Layer")
    print("="*50)

    batch_size, seq_len, input_dim = 2, 10, 2
    output_dim = 768

    # Test with activation and dropout
    emb = embedding(input_channel=input_dim, output_channel=output_dim,
                    bias=False, activation=nn.ReLU, drop=0.1)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = emb(x)

    assert out.shape == (batch_size, seq_len, output_dim), \
        f"Expected shape {(batch_size, seq_len, output_dim)}, got {out.shape}"
    print(f"[PASS] Input shape: {x.shape} -> Output shape: {out.shape}")

    # Test as plain Linear (no activation, no dropout)
    emb_linear = embedding(input_channel=input_dim, output_channel=output_dim,
                           bias=False, activation=None, drop=0.0)
    out_linear = emb_linear(x)
    assert out_linear.shape == (batch_size, seq_len, output_dim)
    print("[PASS] Plain linear embedding works")

    print("[PASS] Embedding tests passed!")
    return True


def test_positional_encoding():
    """Test positional encoding."""
    print("\n" + "="*50)
    print("Testing Positional Encoding")
    print("="*50)

    d_model = 768

    # Test 1D PE
    pos_1d = np.arange(0, 100)
    pe_1d = positional_encoding(pos_1d, d_model, approach='sine')
    assert pe_1d.PE.shape == (1, 100, d_model), \
        f"Expected PE shape (1, 100, {d_model}), got {pe_1d.PE.shape}"
    print(f"[PASS] 1D PE shape: {pe_1d.PE.shape}")

    # Test adding PE to input
    batch_size, seq_len = 2, 100
    x = torch.randn(batch_size, seq_len, d_model)
    out = pe_1d(x)
    assert out.shape == x.shape
    print(f"[PASS] PE addition works: {x.shape} + {pe_1d.PE.shape} -> {out.shape}")

    # Test 2D PE
    # np.mgrid[0:H, 0:W] creates shape [2, H, W] - 2 coordinate arrays
    H, W = 64, 64
    pos_2d = np.mgrid[0:H, 0:W]  # [2, H, W]
    pe_2d = positional_encoding(pos_2d, d_model, approach='sine')
    expected_seq_len = H * W
    assert pe_2d.PE.shape == (1, expected_seq_len, d_model), \
        f"Expected PE shape (1, {expected_seq_len}, {d_model}), got {pe_2d.PE.shape}"
    print(f"[PASS] 2D PE shape: {pe_2d.PE.shape}")

    print("[PASS] Positional encoding tests passed!")
    return True


def test_multi_head_self_attn():
    """Test multi-head self attention."""
    print("\n" + "="*50)
    print("Testing Multi-Head Self Attention")
    print("="*50)

    batch_size, seq_len, d_model = 2, 100, 768
    num_heads = 8

    attn = multi_head_self_attn(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # Test without mask
    # omega = torch.ones(batch_size, seq_len)
    # out = attn(x, omega)
    out = attn(x)
    assert out.shape == x.shape, \
        f"Expected shape {x.shape}, got {out.shape}"
    print(f"[PASS] Self-attention output shape: {out.shape}")

    # Test with mask (some positions masked)
    # omega_masked = torch.ones(batch_size, seq_len)
    # omega_masked[:, ::2] = 0  # Mask every other position
    # out_masked = attn(x, omega_masked)
    # assert out_masked.shape == x.shape
    # print("[PASS] Self-attention with mask works")

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    assert attn.W_QKV.weight.grad is not None
    print("[PASS] Gradient flow works")

    print("[PASS] Multi-head self attention tests passed!")
    return True


def test_multi_head_cross_attn():
    """Test multi-head cross attention."""
    print("\n" + "="*50)
    print("Testing Multi-Head Cross Attention")
    print("="*50)

    batch_size = 2
    seq_len_q, seq_len_kv = 100, 50
    d_model = 768
    num_heads = 8

    cross_attn = multi_head_cross_attn(d_model, num_heads)

    # Q from PE, K/V from encoder output
    PE_p = torch.randn(batch_size, seq_len_q, d_model)
    O = torch.randn(batch_size, seq_len_kv, d_model)

    out = cross_attn(PE_p, O)
    assert out.shape == PE_p.shape, \
        f"Expected shape {PE_p.shape}, got {out.shape}"
    print(f"[PASS] Cross-attention: Q{PE_p.shape}, K/V{O.shape} -> {out.shape}")

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    assert cross_attn.W_Q.weight.grad is not None
    print("[PASS] Gradient flow works")

    print("[PASS] Multi-head cross attention tests passed!")
    return True


def test_encoder_block():
    """Test encoder block."""
    print("\n" + "="*50)
    print("Testing Encoder Block")
    print("="*50)

    batch_size, seq_len, d_model = 2, 100, 768
    num_heads = 8

    enc = encoder_block(d_model, num_heads, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    # omega = torch.ones(batch_size, seq_len)

    # out = enc(x, omega)
    out = enc(x)
    assert out.shape == x.shape
    print(f"[PASS] Encoder output shape: {out.shape}")

    # Test with different sequence lengths
    x_short = torch.randn(batch_size, 50, d_model)
    # omega_short = torch.ones(batch_size, 50)
    # out_short = enc(x_short, omega_short)
    out_short = enc(x_short)
    assert out_short.shape == x_short.shape
    print("[PASS] Encoder handles variable sequence lengths")

    print("[PASS] Encoder block tests passed!")
    return True


def test_decoder_block():
    """Test decoder block."""
    print("\n" + "="*50)
    print("Testing Decoder Block")
    print("="*50)

    batch_size = 2
    seq_len_kv = 50
    d_model = 768
    num_heads = 8
    W, H = 32, 32  # Use smaller W, H for faster tests
    seq_len_full = W * H  # 1024

    # Test LR decoder (with self-attention, NO refine module)
    dec_lr = decoder_block(d_model, num_heads, W, H, dropout=0.1)
    PE_p = torch.randn(batch_size, seq_len_full, d_model)
    O = torch.randn(batch_size, seq_len_kv, d_model)
    # omega = torch.ones(batch_size, seq_len_full)

    # out_lr = dec_lr(PE_p, O, omega, LR=True)
    out_lr = dec_lr(PE_p, O, LR=True)
    assert out_lr.shape == PE_p.shape
    print(f"[PASS] LR decoder output shape: {out_lr.shape}")

    # Test HR decoder (without self-attention, WITH refine module)
    # seq_len_q must match W*H for proper reshape in refine_module
    dec_hr = decoder_block(d_model, num_heads, W, H, dropout=0.1)
    PE_p_hr = torch.randn(batch_size, seq_len_full, d_model)
    O_hr = torch.randn(batch_size, seq_len_kv, d_model)

    out_hr = dec_hr(PE_p_hr, O_hr, LR=False)
    assert out_hr.shape == PE_p_hr.shape
    print(f"[PASS] HR decoder output shape: {out_hr.shape}")

    print("[PASS] Decoder block tests passed!")
    return True


def test_refine_module():
    """Test refine module."""
    print("\n" + "="*50)
    print("Testing Refine Module")
    print("="*50)

    batch_size = 2
    W, H = 32, 32  # Small for testing
    d_model = 768
    seq_len = W * H

    refiner = refine_module(W, H, d_model)
    S = torch.randn(batch_size, seq_len, d_model)

    out = refiner(S)
    assert out.shape == S.shape
    print(f"[PASS] Refine module: {S.shape} -> {out.shape}")

    # Test that output is different from input (processing happened)
    assert not torch.allclose(out, S, atol=1e-5)
    print("[PASS] Refine module produces different output (not identity)")

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    assert refiner.predict.weight.grad is not None
    assert refiner.embedding.weight.grad is not None
    print("[PASS] Gradient flow through refine module works")

    print("[PASS] Refine module tests passed!")
    return True


def test_transformer():
    """Test full transformer model."""
    print("\n" + "="*50)
    print("Testing Full Transformer Model")
    print("="*50)

    # Model parameters
    input_channel = 2  # Real + Imaginary k-space
    output_channel = 768
    d_model = 768
    N = 2  # Number of layers (small for testing)
    num_heads = 8
    W, H = 32, 32  # Small image for testing
    seq_len = W * H

    # Test input
    batch_size = 1
    ksp = torch.randn(batch_size, seq_len, input_channel)
    
    # Create model with proper 2D position array [2, H, W]
    pos = np.mgrid[0:H, 0:W]
    omega = torch.ones(1, seq_len)
    # Mask some positions
    omega[:, ::2] = 0
    model = transformer(
        input_channel=input_channel,
        output_channel=output_channel,
        bias=False,
        pos=pos,
        d_model=d_model,
        omega=omega,
        N=N,
        num_heads=num_heads,
        W=W,
        H=H,
        activition=None,  # Plain linear for input embedding
        approach='sine',
        dropout_emb=0.0,
        dropout_enc=0.0,
        dropout_dec=0.0
    )
    
    # Forward pass
    ksp_us = torch.masked_select(ksp, omega.unsqueeze(-1).bool()).view(batch_size, -1, input_channel)
    output = model(ksp_us)

    # Expected output: [batch_size, seq_len, 2] (real + imag k-space)
    assert output.shape == (batch_size, seq_len, 2), \
        f"Expected shape {(batch_size, seq_len, 2)}, got {output.shape}"
    print(f"[PASS] Transformer output shape: {output.shape}")

    # Test gradient flow
    loss = output.sum()
    loss.backward()
    assert model.predict.weight.grad is not None
    print("[PASS] Gradient flow through full model works")

    # Test with different batch sizes
    ksp_single = torch.randn(1, seq_len, input_channel)
    ksp_single_us = torch.masked_select(ksp_single, omega.unsqueeze(-1).bool()).view(1, -1, input_channel)
    # omega_single = torch.ones(1, seq_len)
    output_single = model(ksp_single_us)
    assert output_single.shape == (1, seq_len, 2)
    print("[PASS] Model handles batch_size=1")

    print("[PASS] Full transformer tests passed!")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING ALL UNIT TESTS FOR TRANSFORMER_MRI")
    print("="*60)

    tests = [
        ("Embedding", test_embedding),
        ("Positional Encoding", test_positional_encoding),
        ("Multi-Head Self Attention", test_multi_head_self_attn),
        ("Multi-Head Cross Attention", test_multi_head_cross_attn),
        ("Encoder Block", test_encoder_block),
        ("Decoder Block", test_decoder_block),
        ("Refine Module", test_refine_module),
        ("Full Transformer", test_transformer),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name} test FAILED with error:")
            print(f"   {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[ALL PASS] All tests passed!")
        return True
    else:
        print(f"\n[WARN] {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
