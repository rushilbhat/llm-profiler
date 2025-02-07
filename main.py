from model import Model

def main():
    # Common hyperparameters (the same for all phases)
    h = 16384
    V = 126000
    L = 126
    n_h = 128
    n_kv = 8
    d_ff = 53248

    model = Model(h, V, L, n_h, n_kv, d_ff)

    phases = {
        "b1k_s4k": {
            "b": 1000,
            "s": 4096,
            "rounds": 62
        },
        "b1k_s8k": {
            "b": 1000,
            "s": 8192,
            "rounds": 350311
        },
        "b2k_s8k": {
            "b": 2000,
            "s": 8192,
            "rounds": 776978
        }
    }

    all_phases_fwd = 0.0
    all_phases_bwd = 0.0
    all_phases_total = 0.0
    all_phases_linear = 0.0

    for phase_name, phase in phases.items():
        print(f"\nPhase: {phase_name}")

        b = phase["b"]
        s = phase["s"]
        rounds = phase["rounds"]

        # Re-compile the operations for the new (b, s)
        model.compile(b, s)

        # Compute FLOPs
        flops_dict = model.get_training_flops(rounds=rounds)
        print(f"Forward FLOPs:  {flops_dict['fwd_flops']}")
        print(f"Backward FLOPs: {flops_dict['bwd_flops']}")
        print(f"Total FLOPs:    {flops_dict['total_flops']}")

        # Accumulate totals
        all_phases_fwd += flops_dict["fwd_flops"]
        all_phases_bwd += flops_dict["bwd_flops"]
        all_phases_total += flops_dict["total_flops"]
        all_phases_linear += flops_dict["total_linear_flops"]

    print("\nTotal across all phases")
    print(f"Forward FLOPs:       {all_phases_fwd}")
    print(f"Backward FLOPs:      {all_phases_bwd}")
    print(f"Total FLOPs:         {all_phases_total}")
    print(f"Total Linear FLOPs:  {all_phases_linear}")

if __name__ == "__main__":
    main() 