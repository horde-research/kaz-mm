
# --- Configuration ---
# You can run this script using command-line arguments:
# python upload_hf.py \
#   --base_model "/workspace/train_data_folder/lora128_vision_layer_cosine_2e-4_bs72_ep1_0412_145218" \
#   --lora_path "/workspace/train_data_folder/RL_STAGE_lora64_1ep_1112_094354" \
#   --hub_repo_id "horde-research/KazVision"
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastVisionModel
from peft import PeftModel
from huggingface_hub import HfFolder

# --- Configuration ---
# You can run this script using command-line arguments:
# python upload_hf.py \
#   --base_model "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit" \
#   --lora_path "./my_lora_adapter" \
#   --hub_repo_id "your-username/my-merged-model"

def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Merge LoRA weights and push to Hugging Face Hub.")
    parser.add_argument(
        "--base_model", 
        type=str, 
        required=True,
        help="The Hugging Face ID of the base model (e.g., unsloth/Qwen2-VL-2B-Instruct-bnb-4bit)"
    )
    parser.add_argument(
        "--lora_path", 
        type=str, 
        required=True,
        help="Local path to the saved LoRA adapter folder (e.g., ./my_lora_adapter_dir)"
    )
    parser.add_argument(
        "--hub_repo_id", 
        type=str, 
        required=True,
        help="The target Hugging Face repository ID (e.g., your-username/my-merged-model)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length for the model (default: 4096)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded repository private"
    )
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Check for Authentication Token
    token = HfFolder.get_token()
    if not token:
        print("🚨 Hugging Face token not found. Please run 'huggingface-cli login' first.")
        return

    print(f"\n[1/5] Loading Base Vision Model with Unsloth: {args.base_model}...")
    
    # --- Load Base Model and Tokenizer using Unsloth ---
    try:
        # Load the base model using FastVisionModel
        base_model, tokenizer = FastVisionModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            dtype=None,          # Automatically uses bfloat16/float16
            load_in_4bit=True,   # Load in 4-bit quantization
        )
        print(f"✓ Base model loaded successfully")
        print(f"  Model dtype: {base_model.dtype}")
        print(f"  Device: {base_model.device}")
    except Exception as e:
        print(f"❌ Error loading base model or tokenizer with Unsloth: {e}")
        return

    print(f"\n[2/5] Loading LoRA Adapter from: {args.lora_path}...")
    
    # 2. Load the LoRA adapter on top of the base model
    try:
        model_with_lora = PeftModel.from_pretrained(
            base_model,
            args.lora_path,
        )
        print(f"✓ LoRA adapter loaded successfully")
    except Exception as e:
        print(f"❌ Error loading PEFT adapter: {e}")
        return

    print("\n[3/5] Merging LoRA weights into the base model...")
    
    # 3. Merge and Unload
    try:
        merged_model = model_with_lora.merge_and_unload()
        print(f"✓ LoRA weights merged successfully")
    except Exception as e:
        print(f"❌ Error merging LoRA weights: {e}")
        return
    
    print(f"\n[4/5] Saving merged model and tokenizer locally to: {args.hub_repo_id}...")

    # Save locally first
    try:
        merged_model.save_pretrained(
            args.hub_repo_id,
            safe_serialization=True  # Use safetensors format
        )
        tokenizer.save_pretrained(args.hub_repo_id)
        print(f"✓ Model and tokenizer saved locally")
    except Exception as e:
        print(f"❌ Error saving model locally: {e}")
        return

    print(f"\n[5/5] Uploading merged model to Hugging Face Hub: {args.hub_repo_id}...")
    
    # 4. Push to Hugging Face Hub
    try:
        merged_model.push_to_hub(
            args.hub_repo_id,
            token=token,
            private=args.private,
            safe_serialization=True,
            commit_message="Upload merged LoRA model"
        )
        tokenizer.push_to_hub(
            args.hub_repo_id,
            token=token,
            private=args.private,
            commit_message="Upload tokenizer"
        )
        print(f"\n✅ Successfully Merged and Uploaded!")
        print(f"   View at: https://huggingface.co/{args.hub_repo_id}")
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face Hub: {e}")
        return

if __name__ == "__main__":
    main()