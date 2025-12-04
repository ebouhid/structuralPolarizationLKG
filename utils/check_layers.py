from sae_lens import SAE

def check_availability():
    # Option 1: Check the Instruct release you found
    print("--- Checking 'llama-3-8b-it-res-jh' ---")
    try:
        # This will fail if the ID is invalid, or print details
        # We just want to see the available SAE IDs in the error or doc
        SAE.from_pretrained("llama-3-8b-it-res-jh", "blocks.16.hook_resid_post", device="cpu")
        print("SUCCESS: Layer 16 exists!")
    except Exception as e:
        print(f"Layer 16 failed (Expected). Error snippet: {str(e)[:200]}...")

    # Option 2: Check EleutherAI (Base model SAEs)
    print("\n--- Checking 'EleutherAI/sae-llama-3-8b-32x' ---")
    try:
        sae, _, _ = SAE.from_pretrained("EleutherAI/sae-llama-3-8b-32x", "blocks.16.hook_resid_post", device="cpu")
        print("SUCCESS: EleutherAI Layer 16 is available!")
    except Exception as e:
        print(f"EleutherAI failed: {e}")

if __name__ == "__main__":
    check_availability()
