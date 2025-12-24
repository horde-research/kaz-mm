
import os
import json
import torch
import re
import base64
import unicodedata
import datetime
from datasets import Dataset, load_dataset
from unsloth import FastVisionModel, is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig
from openai import OpenAI
from PIL import Image
# ==========================================
# 0. CONFIGURATION & API SETUP
# ==========================================
# Set your OpenAI Key
os.environ["OPENAI_API_KEY"] = ""

# Initialize OpenAI Client
client = OpenAI()

# Model Paths
init_model_path = "/path" 
# # ==========================================
#MODEL LOADING
# # ==========================================

model, tokenizer = FastVisionModel.from_pretrained(
    init_model_path,
    load_in_4bit = True, # 4bit is highly recommended for RL to save memory
    max_seq_length = 2048,
)
# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def normalize_unicode_nfc(path: str) -> str:
    return unicodedata.normalize("NFC", path)

def encode_image(image_path):
    """Encodes a local image to base64 for the OpenAI API."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# ==========================================
# 2. DATA PROCESSING (Modified for GRPO)
# ==========================================

def convert_to_conversation(sample):
    # Extract user message block
    user_msg = sample["messages"][0]["content"]
    # Extract assistant message block
    assistant_msg = sample["messages"][1]["content"]

    # Find instruction text (type == 'text')
    instruction = next(
        (c["text"] for c in user_msg if c["type"] == "text"),
        None
    )

    # Find image path (type == 'image')
    image = next(
        (c["image"] for c in user_msg if c["type"] == "image"),
        None
    )

    # Assistant text
    assistant_text = next(
        (c["text"] for c in assistant_msg if c["type"] == "text"),
        None
    )

    img_path = normalize_unicode_nfc('/workspace/train_data_folder/'+image)
    image_pil = Image.open(img_path).convert("RGB")
    image_pil = image_pil.resize((512, 512))

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": instruction},  # The text part of the prompt
            ],
        },
    ]
    return {"prompt": prompt, "image": image_pil, "image_path": img_path, "answer": None}


# Load Data
rl_data_path = "/workspace/train_data_folder/train_RL.json"
with open(rl_data_path) as f:
    da = json.load(f)

# Convert to Dataset object
hf_dataset = Dataset.from_list(da)
processed = []
for idx, i in enumerate(hf_dataset):
    processed.append(convert_to_conversation(i))
    # if idx == 50:
    #     break
#processed = processed[:10]

hf_dataset = Dataset.from_list(processed)
hf_dataset = hf_dataset.map(
    lambda example: {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize = False,
            add_generation_prompt = True, # Must add assistant
        )
    }
)
LLM_MODEL = "gpt-4.1-mini-2025-04-14"
# # ==========================================
# # 3. REWARD FUNCTION (LLM-as-a-Judge)
# # ==========================================
def visual_alignment_reward_func(prompts, completions, completion_ids, **kwargs):
    """
    Evaluates generated Kazakh captions using GPT-4o as a judge.
    
    Args:
        prompts: List of prompt texts (not used, but required by GRPO signature)
        completions: List of generated completions to evaluate
        completion_ids: List of token IDs (not used, but required by GRPO signature)
        **kwargs: Contains 'image_path' and other dataset columns
    
    Returns:
        List of normalized reward scores (0.0 to 1.0)
    """
    # Extract image_path from kwargs
    image_paths = kwargs.get('image_path', [])
    print(f"Image Paths: {image_paths}")
    if len(image_paths) != len(completions):
        print(f"Warning: image_paths ({len(image_paths)}) != completions ({len(completions)})")
        # Return zero rewards if mismatch
        return [0.0] * len(completions)
    
    rewards = []
    
    for img_path, completion in zip(image_paths, completions):
        
        # 1. Encode Image
        base64_image = encode_image(img_path)
        
        if base64_image is None:
            print(f"Failed to encode image: {img_path}")
            rewards.append(0.0)
            continue

        # 2. Call OpenAI (GPT-4o)
        try:
            # Use the detailed prompt structure
            judge_prompt = (
            "You are an expert evaluator for vision-language systems with deep expertise in Kazakh. "
            "Evaluate the Kazakh response quality for the given image on a 1-10 scale. "
            "CRITERIA: Visual Alignment (4pts): Accuracy of described content, spatial relationships, actions. "
            "Penalize hallucinations and missing critical elements. "
            "Kazakh Quality (4pts): Grammar, natural phrasing, proper morphology (cases, verb forms, word order), cultural appropriateness. "
            "Completeness (2pts): Adequate detail and relevance to the task. "
            "GUIDELINES: Perfect 10s are rare, be rigorous. Accuracy over verbosity. "
            "For instructions/questions evaluate if the response appropriately addresses what was asked. "
            "For captions evaluate descriptive accuracy and completeness. "
            "OUTPUT: Return a JSON object with this structure: "
            '{"score": <1-10>, "reasoning": "<15 words max>"}'
            )
            
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": judge_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Generated Kazakh Caption: {completion}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"}
                            }
                        ]
                    }
                ],
                # Ensure the model returns a predictable format
                response_format={"type": "json_object"}
            )
            
            # 3. Parse and Normalize Score
            result = json.loads(response.choices[0].message.content)
            print(f"Image: {os.path.basename(img_path)} | LLM response: {result}")
            
            # Ensure score is between 1 and 10
            raw_score = float(result.get("score", 1))
            raw_score = max(1.0, min(10.0, raw_score)) 
            
            # Normalize 1-10 score to 0.0-1.0 range: (x - min) / (max - min)
            normalized = (raw_score - 1) / 9
            rewards.append(normalized)
            
            print(f"  Raw score: {raw_score}/10 | Normalized: {normalized:.3f}")

        except Exception as e:
            print(f"Judge API Error for {os.path.basename(img_path)}: {e}")
            rewards.append(0.0) # Assign zero reward on API failure
            
    return rewards


# # Enable LoRA
LORA_RANK = 64
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, 
    finetune_language_layers   = True,   # Train LLM
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = LORA_RANK,
    lora_alpha = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.0,
    bias = "none",
    random_state = 3407,
    use_gradient_checkpointing = "unsloth",
)

# # ==========================================
# # 5. TRAINING
# # ==========================================

training_args = GRPOConfig(
    output_dir = "outputs_rl_kazakh",
    per_device_train_batch_size = 48, # Keep small for VRAM
    gradient_accumulation_steps = 2,
    num_train_epochs = 1,
    learning_rate = 5e-6,           # Very low LR for RL
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    num_generations = 2,            # GRPO: Generate 4 answers per image to compare
    max_prompt_length=2048,
    max_completion_length=2048,
    max_grad_norm = 0.1,
    max_steps=50,
    # save
    save_strategy = "steps",        # Save based on steps
    save_steps = 10,                # Save every 10 steps (~5 checkpoints per epoch)
    save_total_limit = 5,           # Keep last 5 checkpoints (covers most of training)
    logging_steps = 2,      
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = visual_alignment_reward_func,
    args = training_args,
    train_dataset = hf_dataset,
)

print("Starting GRPO Training...")
trainer.train()

# ==========================================
# 6. SAVE
# ==========================================
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
now = datetime.datetime.now()
file_timestamp = now.strftime("%d%m_%H%M%S")
file_name = f"RL_STAGE_lora{LORA_RANK}_kazakh_v1_{file_timestamp}"

model.save_pretrained(file_name)
tokenizer.save_pretrained(file_name)
print("RL Training Finished & Saved!")

