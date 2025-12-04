import torch
import os
import json
from sae_lens import SAE
from transformer_lens import HookedTransformer
import gc

def verify():
    # Force garbage collection to clear previous run artifacts
    gc.collect()
    torch.cuda.empty_cache()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Verification Config: Device={device}, Dtype=float16 ---")

    print("1. Loading SAE (Manual Local Load)...")
    base_path = "data/sae/goodfire_l19/blocks.19.hook_resid_post"
    cfg_path = os.path.join(base_path, "cfg.json")
    weight_path = os.path.join(base_path, "sae_weights.pt")
    
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config at {cfg_path}")
        
    with open(cfg_path, 'r') as f:
        cfg_dict = json.load(f)
    
    # Force config to match our hardware constraints
    cfg_dict['device'] = device
    cfg_dict['dtype'] = "float16" 
    
    # Instantiate and Load
    sae = SAE.from_dict(cfg_dict)
    state_dict = torch.load(weight_path, map_location=device)
    
    # Cast weights to half precision to match config
    state_dict = {k: v.to(dtype=torch.float16) for k, v in state_dict.items()}
    sae.load_state_dict(state_dict)
    sae.to(device, dtype=torch.float16) # Explicit move
    
    print(f"   Success! SAE loaded on {sae.device}")

    print("\n2. Loading Model (Meta Llama 3.1)...")
    # CRITICAL FIX: dtype=torch.float16 prevents RAM explosion
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device=device,
        fold_ln=False,
        dtype=torch.float16 
    )
    print(f"   Success! Model loaded on {model.cfg.device} using {model.cfg.dtype}")

    print("\n3. Checking Compatibility...")
    assert model.cfg.d_model == sae.cfg.d_in, \
        f"Mismatch! Model: {model.cfg.d_model} vs SAE: {sae.cfg.d_in}"
        
    print("   Dimensions match. System is ready.")

if __name__ == "__main__":
    verify()
