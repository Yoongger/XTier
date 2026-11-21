"""Checkpoint expert formats: which parameters the checkpoint's own MoE method creates, which of them must be loaded
before the layer can go to the CPU engine, and how they become the engine's inputs (w13 [E, 2I, H/2], w2 [E, H, I/2],
block scales s13 / s2 as bytes, global scales g13 [E, 2] / g2 [E]).
"""

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)


class Format:
    name = ''          # engine scale format
    required = {}      # parameter -> 'w13' (needs the w1 and w3 shards, or the fused w13) or 'w2', per expert
    bulk = ()          # the parameters behind w13, w2, s13, s2: bytes as the checkpoint stores them (direct.py)
    derived = ()       # parameters the method derives from the empty tensors at creation (dropped)
    # The required parameters are released once the engine has its copy; small ones (input scales) stay.

    def engine_inputs(self, layer, up_first: bool):
        raise NotImplementedError

    def globals(self, layer):
        """g13 [E, 2] and g2 [E] (float32), without touching the bulk parameters."""
        raise NotImplementedError


def shard_complete(kind: str, shards: set) -> bool:
    return 'w2' in shards if kind == 'w2' else ('w13' in shards or {'w1', 'w3'} <= shards)


def _gate_up(t: torch.Tensor, up_first: bool) -> torch.Tensor:
    """[E, 2I, ...] in [gate; up] row order (the loader stored [up; gate] when the GPU method asked for it)."""
    if not up_first:
        return t.contiguous()
    half = t.shape[1] // 2
    return torch.cat([t[:, half:], t[:, :half]], dim=1).contiguous()


def _dequant_fp8_blocks(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """FP8 (E4M3) block-quantized weights -> BF16: weight * scale_inv per block.

    The scale is the reciprocal (SGLang's *_weight_scale_inv) and the block shape comes from the scale
    tensor's shape (the checkpoint's blocks are 128x128 here). BF16 keeps every E4M3 value exactly; only the
    scaled product is rounded to BF16 (the engine's FP4 path rounds the same way).
    """
    E, R, C = weight.shape
    sr, sc = scale_inv.shape[-2], scale_inv.shape[-1]
    bn, bk = (R + sr - 1) // sr, (C + sc - 1) // sc
    dev = weight.device
    out = torch.empty((E, R, C), dtype=torch.bfloat16, device=dev)
    for r0 in range(0, R, bn):
        for c0 in range(0, C, bk):
            rs = min(bn, R - r0)
            cs = min(bk, C - c0)
            blk = weight[:, r0:r0 + rs, c0:c0 + cs].to(torch.float32)
            si = scale_inv[:, r0 // bn, c0 // bk].to(torch.float32).reshape(E, 1, 1)
            out[:, r0:r0 + rs, c0:c0 + cs] = (blk * si).to(torch.bfloat16)
    return out


class Fp8BlockExpertsAsBf16(Format):
    """FP8 (E4M3) block-quantized experts (BF16 block scales, e.g. 128x128) for the CPU engine.

    The engine's FP4 formats (nvfp4 / mxfp4) cannot hold these weights; the engine gained a BF16 expert
    format (scale_format=2) for them, and this class hands it the same weights dequantized to BF16.
    FP8 -> BF16 keeps every weight value exactly (E4M3 fits in BF16); a 4-bit NVFP4 conversion instead
    would cost 9.5% of the MoE output (measured on this checkpoint's layer 0)."""

    name = 'bf16'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2',
                'w13_weight_scale_inv': 'w13', 'w2_weight_scale_inv': 'w2'}

    def engine_inputs(self, layer, up_first):
        dev = layer.w13_weight.data.device
        w13 = _gate_up(layer.w13_weight.data, up_first).to(dev)
        w2 = layer.w2_weight.data.to(dev)
        s13 = layer.w13_weight_scale_inv.data.to(dev)
        s2 = layer.w2_weight_scale_inv.data.to(dev)
        w13b = _dequant_fp8_blocks(w13, s13)
        w2b = _dequant_fp8_blocks(w2, s2)
        logger.info('xtier: %d FP8 experts dequantized to BF16 for the CPU engine (lossless)', w13.shape[0])
        dummy = torch.ones((1,), dtype=torch.uint8)
        return w13b, w2b, dummy, dummy, *self.globals(layer)

    def globals(self, layer):
        E = layer.w13_weight.data.shape[0]
        return torch.ones((E, 2), dtype=torch.float32), torch.ones((E,), dtype=torch.float32)


class Fp8BlockExpertsAsNvFp4(Format):
    """FP8 (E4M3) block-quantized experts -> NVFP4 for the CPU engine (the engine's native format).

    FP8 -> BF16 is lossless; BF16 -> NVFP4 is the 4-bit step (9.5% of the MoE output, measured on this
    checkpoint). The engine's nvfp4 path is what the author's Qwen3.8 production setup uses (180 hot slots
    per layer, hit rate 0.92); the bf16 format keeps the precision but fits a quarter of the hot slots."""

    name = 'nvfp4'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2',
                'w13_weight_scale_inv': 'w13', 'w2_weight_scale_inv': 'w2'}

    def engine_inputs(self, layer, up_first):
        dev = layer.w13_weight.data.device
        w13 = _gate_up(layer.w13_weight.data, up_first).to(dev)
        w2 = layer.w2_weight.data.to(dev)
        s13 = layer.w13_weight_scale_inv.data.to(dev)
        s2 = layer.w2_weight_scale_inv.data.to(dev)
        E, I2, H = w13.shape
        I = I2 // 2
        q13 = torch.empty((E, I2, H // 2), dtype=torch.uint8)
        s13b = torch.empty((E, I2, H // 16), dtype=torch.uint8)
        q2 = torch.empty((E, H, I // 2), dtype=torch.uint8)
        s2b = torch.empty((E, H, I // 16), dtype=torch.uint8)
        g13 = torch.empty((E, 2), dtype=torch.float32)
        g2 = torch.empty((E,), dtype=torch.float32)
        for e in range(E):
            m = _dequant_fp8_blocks(w13[e:e + 1], s13[e:e + 1])[0].float()
            for j in (0, 1):   # gate rows, up rows: a global scale each
                rows = slice(j * I, (j + 1) * I)
                q, sc, g = nvfp4_encode(m[rows])
                q13[e, rows], s13b[e, rows], g13[e, j] = q.to(torch.uint8).cpu(), sc.view(torch.uint8).cpu(), g.item()
            q, sc, g = nvfp4_encode(_dequant_fp8_blocks(w2[e:e + 1], s2[e:e + 1])[0].float())
            q2[e], s2b[e], g2[e] = q.to(torch.uint8).cpu(), sc.view(torch.uint8).cpu(), g.item()
        logger.info('xtier: %d FP8 experts quantized to NVFP4 for the CPU engine (4-bit: 9.5%% of the MoE output)',
                    E)
        return q13, q2, s13b, s2b, g13, g2


class ModelOptNvFp4(Format):
    """ModelOpt NVFP4 (and W4A16_NVFP4): E2M1 weights, E4M3 scales per 16, FP32 scale per expert and projection."""
    name = 'nvfp4'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2', 'w13_weight_scale': 'w13', 'w2_weight_scale': 'w2',
                'w13_weight_scale_2': 'w13', 'w2_weight_scale_2': 'w2'}
    bulk = ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale')
    derived = ('w13_blockscale_swizzled', 'w2_blockscale_swizzled')

    def engine_inputs(self, layer, up_first):
        return (_gate_up(layer.w13_weight.data, up_first), layer.w2_weight.data.contiguous(),
                _gate_up(layer.w13_weight_scale.data.view(torch.uint8), up_first),
                layer.w2_weight_scale.data.view(torch.uint8).contiguous()) + self.globals(layer)

    def globals(self, layer):
        g13 = layer.w13_weight_scale_2.data.float()   # per-tensor scales always stay [gate, up]
        if g13.dim() == 1:
            g13 = g13[:, None].expand(-1, 2)
        return g13.contiguous(), layer.w2_weight_scale_2.data.float().contiguous()


class CompressedTensorsNvFp4(Format):
    """compressed-tensors NVFP4 W4A4 (MiniMax M3.1's routed experts): packed E2M1 weights, E4M3 scales per 16 and an
    FP32 global scale per expert and projection that divides (ModelOpt's scale_2 is its reciprocal). The activations'
    FP4 global scales are not needed: the engine and XTier's GPU kernels multiply BF16 activations."""
    name = 'nvfp4'
    required = {'w13_weight_packed': 'w13', 'w2_weight_packed': 'w2', 'w13_weight_scale': 'w13', 'w2_weight_scale': 'w2',
                'w13_weight_global_scale': 'w13', 'w2_weight_global_scale': 'w2'}
    bulk = ('w13_weight_packed', 'w2_weight_packed', 'w13_weight_scale', 'w2_weight_scale')

    def engine_inputs(self, layer, up_first):
        return (_gate_up(layer.w13_weight_packed.data, up_first), layer.w2_weight_packed.data.contiguous(),
                _gate_up(layer.w13_weight_scale.data.view(torch.uint8), up_first),
                layer.w2_weight_scale.data.view(torch.uint8).contiguous()) + self.globals(layer)

    def globals(self, layer):
        g13 = layer.w13_weight_global_scale.data.float().reciprocal()   # per-tensor scales always stay [gate, up]
        if g13.dim() == 1:
            g13 = g13[:, None].expand(-1, 2)
        return g13.contiguous(), layer.w2_weight_global_scale.data.float().reciprocal().contiguous()


def _ue8m0_bytes(t: torch.Tensor) -> torch.Tensor:
    """UE8M0 block scales as bytes: SGLang keeps them as float32 powers of two (float8_e8m0fnu with aiter / XPU); the
    exponent field of a float32 2^(e-127) is e, including 2^-127 (exponent field 0)."""
    if t.dtype == torch.uint8:
        return t.contiguous()
    if t.dtype == torch.float8_e8m0fnu:
        return t.view(torch.uint8).contiguous()
    if t.dtype == torch.float32:
        bits = t.contiguous().view(torch.int32)
        if bool(((bits & 0x7FFFFF) != 0).logical_and((bits >> 23) != 0).any()):
            raise ValueError('xtier: an MXFP4 block scale is not a power of two')
        return ((bits >> 23) & 0xFF).to(torch.uint8)
    raise TypeError(f'xtier: MXFP4 block scales of dtype {t.dtype}')


class Fp8Fp4Experts(Format):
    """DeepSeek V4 MXFP4 experts (quantization config fp8, expert_dtype fp4): E2M1 weights, UE8M0 scales per 32, no
    global scales."""
    name = 'mxfp4'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2', 'w13_weight_scale_inv': 'w13', 'w2_weight_scale_inv': 'w2'}
    bulk = ('w13_weight', 'w2_weight', 'w13_weight_scale_inv', 'w2_weight_scale_inv')

    def engine_inputs(self, layer, up_first):
        return (_gate_up(layer.w13_weight.data.view(torch.uint8), up_first), layer.w2_weight.data.view(torch.uint8).contiguous(),
                _gate_up(_ue8m0_bytes(layer.w13_weight_scale_inv.data), up_first),
                _ue8m0_bytes(layer.w2_weight_scale_inv.data)) + self.globals(layer)

    def globals(self, layer):
        E = layer.w13_weight.data.shape[0]
        return torch.ones((E, 2), dtype=torch.float32), torch.ones((E,), dtype=torch.float32)


E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)   # midpoints of the E2M1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6


def nvfp4_encode(w: torch.Tensor):
    """ModelOpt NVFP4 of a matrix w [rows, K], K a multiple of 16: (E2M1 pairs [rows, K/2], the even element in the low
    nibble; E4M3 block scales [rows, K/16] as bytes; the FP32 global scale amax / (6 * 448)), w ~ e2m1 x scale x global.
    Rounding to the nearest E2M1 value of w / (scale x global)."""
    w = w.float()
    g = (w.abs().amax() / (6 * 448)).clamp_min(1e-12)
    blocks = w.unflatten(-1, (-1, 16))
    s = (blocks.abs().amax(-1) / (6 * g)).to(torch.float8_e4m3fn)   # at most 448: the global scale's choice
    q = (blocks / (s.float() * g).clamp_min(1e-30).unsqueeze(-1)).flatten(-2)   # a zero scale dequantizes to 0
    code = torch.bucketize(q.abs().clamp(max=6.0), torch.tensor(E2M1_BOUNDS, device=w.device)).to(torch.uint8)
    code |= ((q < 0) & (code > 0)).to(torch.uint8) << 3
    return code[:, 0::2] | (code[:, 1::2] << 4), s.view(torch.uint8), g


class Bf16AsNvFp4(Format):
    """A draft model's BF16 experts (GLM-5.3's NextN layer in NVIDIA's ModelOpt checkpoint, excluded from its NVFP4)
    quantized to ModelOpt NVFP4 at the hand-off, so the CPU engine runs them as it runs the target's: the draft's
    precision only, the target model verifies every token the draft proposes. Per expert on the GPU when there is one;
    a global scale per expert and projection."""
    name = 'nvfp4'
    required = {'w13_weight': 'w13', 'w2_weight': 'w2'}

    def engine_inputs(self, layer, up_first):
        t0 = time.perf_counter()
        w13, w2 = layer.w13_weight.data, layer.w2_weight.data   # [E, 2I, H], [E, H, I] BF16
        E, I2, H = w13.shape
        dev = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else w13.device
        q13 = torch.empty((E, I2, H // 2), dtype=torch.uint8)
        s13 = torch.empty((E, I2, H // 16), dtype=torch.uint8)
        q2 = torch.empty((E, H, I2 // 4), dtype=torch.uint8)
        s2 = torch.empty((E, H, I2 // 32), dtype=torch.uint8)
        g13 = torch.empty((E, 2), dtype=torch.float32)
        g2 = torch.empty((E,), dtype=torch.float32)
        for e in range(E):
            m = _gate_up(w13[e:e + 1], up_first)[0].to(dev)
            for j in range(2):   # gate rows, up rows: a global scale each
                rows = slice(j * I2 // 2, (j + 1) * I2 // 2)
                q, s, g = nvfp4_encode(m[rows])
                q13[e, rows], s13[e, rows], g13[e, j] = q.cpu(), s.cpu(), g.item()
            q, s, g = nvfp4_encode(w2[e].to(dev))
            q2[e], s2[e], g2[e] = q.cpu(), s.cpu(), g.item()
        logger.info('xtier: %d BF16 draft experts quantized to NVFP4 for the CPU engine in %.1f s (the draft\'s '
                    'precision only: the target model verifies every token)', E, time.perf_counter() - t0)
        return q13, q2, s13, s2, g13, g2


def detect(gpu_method, draft=False) -> Format:
    name = type(gpu_method).__name__
    if name == 'ModelOptNvFp4FusedMoEMethod':
        return ModelOptNvFp4()
    if name == 'Fp8MoEMethod' and getattr(gpu_method, 'is_fp4_expert', False):
        return Fp8Fp4Experts()
    if name == 'Fp8MoEMethod':
        # FP8 E4M3 experts with BF16 block scales (Qwen3.8 FP8): the CPU engine takes them as BF16 (lossless)
        # or as NVFP4 (4-bit, the engine's native format: four times smaller, four times more hot slots).
        if os.environ.get('XTIER_FP8_EXPERT_FORMAT', 'nvfp4').lower() == 'bf16':
            return Fp8BlockExpertsAsBf16()
        return Fp8BlockExpertsAsNvFp4()
    if name == 'Mxfp4FlashinferCutlassMoEMethod' and getattr(getattr(gpu_method, '_fp8', None), 'is_fp4_expert', False):
        return Fp8Fp4Experts()   # SGLang's SM90 / SM120 choice for these experts: Fp8MoEMethod's parameters, E8M0 scales
    if name == 'CompressedTensorsFusedMoEMethod':
        return CompressedTensorsNvFp4()   # the scheme sits on the layer: checked once the parameters exist
    if name == 'UnquantizedFusedMoEMethod' and draft:
        return Bf16AsNvFp4()   # a target model's BF16 experts stay with SGLang: no precision change there
    raise NotImplementedError(f'xtier: no CPU engine format for the MoE method {name}')
