from math import prod

class Operation:
    def __init__(self):
        self.fwd_flops = 0.0
        self.bwd_flops = 0.0
        
    @property
    def total_flops(self):
        return self.fwd_flops + self.bwd_flops

class Linear(Operation):
    def __init__(self, input: tuple[int, ...], out_dim: int):
        super().__init__()
        # Input: [b, s, h]
        # Params: [h, out_dim]
        # Output: [b, s, out_dim]
        self.fwd_flops = 2.0 * prod(input) * out_dim
        self.bwd_flops = 2.0 * self.fwd_flops

class Embedding(Operation):
    def __init__(self):
        super().__init__()
        # We often approximate as 0 or negligible
        self.fwd_flops = 0.0
        self.bwd_flops = 0.0

class RMSNorm(Operation):
    def __init__(self, input: tuple[int, ...]):
        # Assuming normalised shape is input[-1]
        super().__init__()
        # Input: [b, s, h]
        # Params: [h]
        # Output: [b, s, h]
        self.fwd_flops = 4.0 * prod(input)
        # PyTorch autograd breaks this down into several ops:
        # MulBackward0 
        # Incoming grad [b,s,h]; grad for broadcasted gain [b,s,h] = [b,s,h] * [b,s,h] elementwise, grad for gain [h,] = reduce across b,s; grad for fraction [b,s,h] = [b,s,h] * [b,s,h] elementwise => 3bsh flops
        #
        # MulBackward0
        # Incoming grad [b,s,h]; grad for numerator [b,s,h] = [b,s,h] * [b,s,h] elementwise; grad for broadcasted denominator [b,s,h] = [b,s,h] * [b,s,h] elementwise; 
        # grad for denominator = (1,) = sum grad for broadcasted denominator => 3bsh flops
        #
        # RsqrtBackward0
        # Negligible flops since we're working with tensor of shape (1,)
        #
        # AddBackward1
        # Negligible flops since we're working with tensor of shape (1,)
        #
        # MeanBackward1
        # Incoming grad (1,); grad for squared tensor [b,s,h] = expand incoming grad from (1,) to [b,s,h] then divide tensor by scalar n => bsh flops
        #
        # PowBackward0
        # Incoming grad [b,s,h]; grad for input tensor [b,s,h] = raise input tensor [b,s,h] to power of 1, then multiply coefficient, then multiply incoming grad => 2bsh flops
        # Total ~9bsh flops per RMSNorm backward
        self.bwd_flops = 9.0 * prod(input)

class Softmax(Operation):
    def __init__(self, input: tuple[int, ...]):
        super().__init__()
        # Input Scores: [b, n_h, s, s]
        # Output: [b, n_h, s, s]
        self.fwd_flops = 5.0 * prod(input)
        self.bwd_flops = 4.0 * prod(input)

class Matmul(Operation):
    def __init__(self, input1: tuple[int, ...], input2: tuple[int, ...]):
        super().__init__()
        # For matrix multiplication of shapes [b, s, m, k] @ [b, s, k, n]
        self.fwd_flops = 2.0 * prod(input1) * input2[-1]
        self.bwd_flops = 2.0 * self.fwd_flops

class Silu(Operation):
    def __init__(self, input: tuple[int, ...]):
        super().__init__()
        # Input: [b, s, dim]
        self.fwd_flops = 5.0 * prod(input)
        # ~4 flops for sigmoid + 5 flops for grad_output * sigmoid * (1 + self * (1 - sigmoid))
        self.bwd_flops = 9.0 * prod(input)

class Residual(Operation):
    def __init__(self, input: tuple[int, ...]):
        super().__init__()
        # Two residual adds per layer
        # Input: [b, s, h], [b, s, h]
        self.fwd_flops = prod(input)
        # No flops, just distribute upstream grad through assignment
        self.bwd_flops = 0.0

class Scale(Operation):
    def __init__(self, input: tuple[int, ...]):
        super().__init__()
        # Input Scores: [b, n_h, s, s]
        self.fwd_flops = prod(input)
        self.bwd_flops = self.fwd_flops

class RotaryEmb(Operation):
    def __init__(self, input: tuple[int, ...]):
        super().__init__()
        # Input Q: [b, s, n_h, d/2] or K: [b, s, n_kv, d/2]
        # Elementwise multiply: 6 flops per element (4 muls + 2 adds)
        self.fwd_flops = 3.0 * prod(input)
        #Elementwise multiply for x_q grad or x_k grad
        self.bwd_flops = self.fwd_flops

class RepeatKV(Operation):
    def __init__(self, input: tuple[int, ...], n_rep: int):
        super().__init__()
        # Input xk or xv: [b, s, n_kv, d] each
        # Summation across repeated dimension
        self.fwd_flops = 0.0  # Negligible in forward pass
        self.bwd_flops = prod(input) * (n_rep - 1)

class Elementwise(Operation):
    def __init__(self, input: tuple[int, ...]):
        super().__init__()
        # Input: [b, s, d_eff], [b, s, d_eff]
        self.fwd_flops = prod(input)
        self.bwd_flops = 2*prod(input)

def create_llama_ops(
    b: int,       # batch size
    s: int,       # sequence length
    h: int,       # model dimension
    V: int,       # vocab size
    L: int,       # number of layers
    n_h: int,     # total # of attention heads
    n_kv: int,    # number of key-value heads (<= n_h)
    d_ff: int,    # feed-forward hidden dimension
):
    d = h / n_h  # head dimension
    ops = []

    ops.append(("embedding", Embedding()))

    # Per-layer operations
    layer_ops = []

    # Attention module
    layer_ops.append(("pre_attn_rmsnorm", RMSNorm((b, s, h))))

    layer_ops.append(("q_proj", Linear((b, s, h), h)))
    layer_ops.append(("k_proj", Linear((b, s, h), n_kv * d)))
    layer_ops.append(("v_proj", Linear((b, s, h), n_kv * d)))
    
    layer_ops.append(("q_rotary", RotaryEmb((b, s, n_h, d))))
    layer_ops.append(("k_rotary", RotaryEmb((b, s, n_kv, d))))
    
    n_rep = n_h / n_kv
    layer_ops.append(("k_repeat_kv", RepeatKV((b, s, n_kv, d), n_rep)))
    layer_ops.append(("v_repeat_kv", RepeatKV((b, s, n_kv, d), n_rep)))
    
    layer_ops.append(("attn_scores", Matmul((b, n_h, s, d), (b, n_h, d, s))))
    layer_ops.append(("attn_scale", Scale((b, n_h, s, s))))
    layer_ops.append(("softmax", Softmax((b, n_h, s, s))))
    layer_ops.append(("attn_v", Matmul((b, n_h, s, s), (b, n_h, s, d))))
    
    layer_ops.append(("out_proj", Linear((b, s, h), h)))
    
    layer_ops.append(("post_attn_residual", Residual((b, s, h))))

    # FFN module
    layer_ops.append(("pre_ffn_rmsnorm", RMSNorm((b, s, h))))

    layer_ops.append(("ffn_up1", Linear((b, s, h), d_ff)))
    layer_ops.append(("ffn_up2", Linear((b, s, d_ff), h)))
    layer_ops.append(("ffn_elemtwise", Elementwise((b, s, d_ff))))
    layer_ops.append(("ffn_act", Silu((b, s, d_ff))))
    layer_ops.append(("ffn_down", Linear((b, s, d_ff), h)))
    
    layer_ops.append(("post_ffn_residual", Residual((b, s, h))))

    # Multiply all layer operations by number of layers
    for name, op in layer_ops:
        op.fwd_flops *= L
        op.bwd_flops *= L
        ops.append((f"all_layers_{name}", op))

    ops.append(("final_norm", RMSNorm((b, s, h))))
    ops.append(("lm_head", Linear((b, s, h), V)))

    return ops

if __name__ == "__main__":
    phases = {
        "b1k_s4k": {
            "b": 1000,
            "s": 4096,
            "h": 16384,
            "V": 126000,
            "L": 126,
            "n_h": 128,
            "n_kv": 8,
            "d_ff": 53248,
            "rounds": 62
        },
        "b1k_s8k": {
            "b": 1000,
            "s": 8192,
            "h": 16384,
            "V": 126000,
            "L": 126,
            "n_h": 128,
            "n_kv": 8,
            "d_ff": 53248,
            "rounds": 350311
        },
        "b2k_s8k": {
            "b": 2000,
            "s": 8192,
            "h": 16384,
            "V": 126000,
            "L": 126,
            "n_h": 128,
            "n_kv": 8,
            "d_ff": 53248,
            "rounds": 776978
        }
    }

    all_phases_fwd = 0.0
    all_phases_bwd = 0.0
    all_phases_total = 0.0
    all_phases_linear_total = 0.0

    for phase_name, phase in phases.items():
        print(f"\nPhase: {phase_name}")
        cfg = {k: v for k, v in phase.items() if k != "rounds"}
        ops = create_llama_ops(**cfg)
        
        # Calculate total FLOPs
        fwd_total = sum(op.fwd_flops for _, op in ops)
        bwd_total = sum(op.bwd_flops for _, op in ops)
        total_flops = fwd_total + bwd_total
        
        # Calculate linear FLOPs using isinstance
        linear_flops = sum((op.fwd_flops + op.bwd_flops) for _, op in ops if isinstance(op, Linear))
        
        # Multiply by number of rounds
        rounds = phase["rounds"]
        fwd_total *= rounds
        bwd_total *= rounds
        total_flops *= rounds
        linear_flops *= rounds
        
        # Add to all phases total
        all_phases_fwd += fwd_total
        all_phases_bwd += bwd_total
        all_phases_total += total_flops
        all_phases_linear_total += linear_flops
        
        print(f"Forward FLOPs: {fwd_total}")
        print(f"Backward FLOPs: {bwd_total}")
        print(f"Total FLOPs: {total_flops}")

    print(f"\nTotal across all phases:")
    print(f"Forward FLOPs: {all_phases_fwd}")
    print(f"Backward FLOPs: {all_phases_bwd}")
    print(f"Total FLOPs: {all_phases_total}")
    print(f"Total Linear FLOPs: {all_phases_linear_total}")
