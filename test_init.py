import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from scripts.analyze_fineweb_trust_ratio import parse_history

def main():
    wandb_root = Path("wandb")
    run_id = "9qnh0b9k"  # the 125m calibrated run
    rows = parse_history(run_id, wandb_root)
    
    # Check step 0 or early steps
    if not rows:
        print("No rows found")
        return
        
    print("Found", len(rows), "rows")
    
    layer_names = [
        "transformer.h.0.mlp.up_proj.weight",
        "transformer.h.0.mlp.down_proj.weight",
        "transformer.h.0.attn.c_proj.weight",   # Out proj
        "transformer.h.0.attn.c_attn.weight"    # qkv
    ]
    
    for row in rows[:5]:  # Look at first few logged steps
        step = row.get("_step", row.get("step", -1))
        print(f"\n--- Step {step} ---")
        for k, v in row.items():
            if "trust_ratio_raw" in k and any(l in k for l in ["mlp.c_fc", "mlp.c_proj", "attn.c_proj", "attn.c_attn"]):
                 print(f"{k}: {v}")
            # The keys might be named differently, let's find the exact names
            if "W_frob" in k or "c_denom" in k:
                if any(x in k for x in ["down_proj", "up_proj", "o_proj", "qkv_proj", "c_proj", "c_attn", "c_fc"]):
                     # print(f"{k}: {v}")
                     pass
                     
        # Just list all keys that have c_denom
        c_denom_keys = [k for k in row.keys() if "c_denom" in k]
        for k in c_denom_keys:
            if "h.0." in k or "h.6." in k:
                 print(f"{k}: {row[k]}")

if __name__ == "__main__":
    main()
