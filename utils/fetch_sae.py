import torch
import json
import os
from huggingface_hub import hf_hub_download

def adapt_goodfire_sae():
    # 1. Setup Paths
    target_folder = "data/sae/goodfire_l19/blocks.19.hook_resid_post"
    os.makedirs(target_folder, exist_ok=True)
    
    print(f"--- Adapting Goodfire SAE to {target_folder} ---")

    # 2. Download or Load Local Weights
    repo_id = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
    filename = "Llama-3.1-8B-Instruct-SAE-l19.pth"
    
    print("Checking for weights...")
    weights_path = hf_hub_download(repo_id, filename=filename)
    
    # 3. Load and Inspect Weights
    print("Loading state dict...")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # SAELens Standard Dictionary
    new_state_dict = {}
    keys = list(state_dict.keys())
    print(f"Found keys: {keys}")

    # --- Handle Goodfire Naming Convention ---
    if "encoder_linear.weight" in state_dict:
        print("Detected Goodfire 'linear' naming convention.")
        
        # Encoder: [d_sae, d_model] -> Transpose to [d_model, d_sae]
        enc_weight = state_dict["encoder_linear.weight"]
        new_state_dict["W_enc"] = enc_weight.T 
        new_state_dict["b_enc"] = state_dict["encoder_linear.bias"]
        
        # Decoder: [d_model, d_sae] -> Transpose to [d_sae, d_model]
        # Note: SAELens expects W_dec to be [d_sae, d_model]
        dec_weight = state_dict["decoder_linear.weight"]
        new_state_dict["W_dec"] = dec_weight.T 
        new_state_dict["b_dec"] = state_dict["decoder_linear.bias"]

        d_sae, d_model = enc_weight.shape 
        
    elif "encoder.weight" in state_dict:
        # Fallback
        print("Detected Goodfire 'standard' naming convention.")
        enc_weight = state_dict["encoder.weight"]
        new_state_dict["W_enc"] = enc_weight.T
        new_state_dict["b_enc"] = state_dict["encoder.bias"]
        new_state_dict["W_dec"] = state_dict["decoder.weight"].T
        new_state_dict["b_dec"] = state_dict["decoder.bias"]
        d_sae, d_model = enc_weight.shape

    else:
        raise ValueError(f"Unknown weight format. Keys found: {keys}")

    print(f"Inferred Dimensions: d_model={d_model}, d_sae={d_sae}")

    # 4. Save Weights
    save_path = os.path.join(target_folder, "sae_weights.pt")
    torch.save(new_state_dict, save_path)
    print(f"Saved adapted weights to {save_path}")

    # 5. Create Config (FIXED)
    sae_config = {
        "architecture": "standard", # <--- CRITICAL FIX
        "d_in": d_model,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hook_name": "blocks.19.hook_resid_post",
        "hook_layer": 19,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_version": "4.0.0",
        "prepend_bos": True,
        "dataset_path": "",
        "context_size": 1024,
        "normalize_activations": False,
    }
    
    with open(os.path.join(target_folder, "cfg.json"), "w") as f:
        json.dump(sae_config, f, indent=4)
        
    print("Saved adapter config to cfg.json")

if __name__ == "__main__":
    adapt_goodfire_sae()
