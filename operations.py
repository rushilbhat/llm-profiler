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
        # For matrix multiplication of shapes [b, n_h, m, k] @ [b, n_h, k, n] - in the case of attention scores m=s, k=d, n=s; in the case of attention values m=s, k=s, n=d
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