from operations import *

class Model:
    def __init__(self, h: int, V: int, L: int, n_h: int, n_kv: int, d_ff: int):
        self.h = h # model dimension
        self.V = V # vocab size
        self.L = L # number of layers
        self.n_h = n_h # total # of attention heads
        self.n_kv = n_kv # number of key-value heads (<= n_h)
        self.d_ff = d_ff # feed-forward hidden dimension
        self.ops = []

    def clear_ops(self):
        """Clear all operations from the model."""
        self.ops = []

    def compile(self, b: int, s: int):
        d = self.h / self.n_h  # head dimension

        # Clear existing operations
        self.clear_ops()

        # Embedding
        self.ops.append(("embedding", Embedding()))

        # Build per-layer operations
        layer_ops = []

        # Attention module
        layer_ops.append(("pre_attn_rmsnorm", RMSNorm((b, s, self.h))))

        layer_ops.append(("q_proj", Linear((b, s, self.h), self.h)))
        layer_ops.append(("k_proj", Linear((b, s, self.h), int(self.n_kv * d))))
        layer_ops.append(("v_proj", Linear((b, s, self.h), int(self.n_kv * d))))

        layer_ops.append(("q_rotary", RotaryEmb((b, s, self.n_h, d))))
        layer_ops.append(("k_rotary", RotaryEmb((b, s, self.n_kv, d))))

        n_rep = self.n_h / self.n_kv
        layer_ops.append(("k_repeat_kv", RepeatKV((b, s, self.n_kv, d), n_rep)))
        layer_ops.append(("v_repeat_kv", RepeatKV((b, s, self.n_kv, d), n_rep)))

        layer_ops.append(("attn_scores", Matmul((b, self.n_h, s, d), (b, self.n_h, d, s))))
        layer_ops.append(("attn_scale", Scale((b, self.n_h, s, s))))
        layer_ops.append(("softmax", Softmax((b, self.n_h, s, s))))
        layer_ops.append(("attn_v", Matmul((b, self.n_h, s, s), (b, self.n_h, s, d))))

        layer_ops.append(("out_proj", Linear((b, s, self.h), self.h)))
        layer_ops.append(("post_attn_residual", Residual((b, s, self.h))))

        # FFN module
        layer_ops.append(("pre_ffn_rmsnorm", RMSNorm((b, s, self.h))))
        layer_ops.append(("ffn_up1", Linear((b, s, self.h), self.d_ff)))
        layer_ops.append(("ffn_up2", Linear((b, s, self.d_ff), self.h)))
        layer_ops.append(("ffn_elemtwise", Elementwise((b, s, self.d_ff))))
        layer_ops.append(("ffn_act", Silu((b, s, self.d_ff))))
        layer_ops.append(("ffn_down", Linear((b, s, self.d_ff), self.h)))
        layer_ops.append(("post_ffn_residual", Residual((b, s, self.h))))

        # Scale per-layer ops by L
        for name, op in layer_ops:
            op.fwd_flops *= self.L
            op.bwd_flops *= self.L
            self.ops.append((f"all_layers_{name}", op))

        # Final norm + LM head
        self.ops.append(("final_norm", RMSNorm((b, s, self.h))))
        self.ops.append(("lm_head", Linear((b, s, self.h), self.V)))

    def get_training_flops(self, rounds: int = 1):
        fwd_total = sum(op.fwd_flops for _, op in self.ops)
        bwd_total = sum(op.bwd_flops for _, op in self.ops)
        total_flops = fwd_total + bwd_total

        # We can also extract linear-only FLOPs:
        linear_total = sum(
            (op.fwd_flops + op.bwd_flops) for _, op in self.ops if isinstance(op, Linear)
        )

        # Multiply by rounds
        return {
            "fwd_flops": fwd_total * rounds,
            "bwd_flops": bwd_total * rounds,
            "total_flops": total_flops * rounds,
            "total_linear_flops": linear_total * rounds
        } 