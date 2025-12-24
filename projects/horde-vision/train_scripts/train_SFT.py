from datasets import Dataset
from PIL import Image
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import unicodedata
import os
import datetime


def normalize_unicode_nfc(path: str) -> str:
    return unicodedata.normalize("NFC", path)
# LOAD DATA
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

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": normalize_unicode_nfc('/workspace/train_data_folder/'+image)},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        },
    ]

    return {"messages": conversation}

import json
sft_data_path = "/workspace/train_data_folder/train_SFT.json"

with open(sft_data_path) as f:
    da = json.load(f)

hf_dataset = Dataset.from_list(da)
processed = []
for i in hf_dataset:
    try:
        processed.append(convert_to_conversation(i))
    except:
        print('nu bivaet i tak')
print(processed[0])
print(len(processed))
print("DATA LOADED!")
init_model_path = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
#init_model_path = "/workspace/train_data_folder/lora64_vision_layer_cosine_2e-4_bs96_ep1_0412_044414"
model, tokenizer = FastVisionModel.from_pretrained(
    init_model_path,
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
LORA_RANK = 64
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,   # I recommend to freeze the vision tower
    finetune_language_layers   = True,    # set False to freeze the text backbone
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = LORA_RANK,
    lora_alpha = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.0,
    bias = "none",
    random_state = 3407,
    use_gradient_checkpointing = "unsloth",  # True or "unsloth" for long context
)


FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = processed,
    args = SFTConfig(
        per_device_train_batch_size = 96,
        #per_device_train_batch_size = 72,
        gradient_accumulation_steps = 4,
        #warmup_steps = 5,
        #max_steps = 1,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        warmup_ratio = 0.1,
        seed = 123,
        output_dir = "outputs_test",
        report_to = "none",     # For Weights and Biases
        #resume_from_checkpoint = True,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

trainer_stats = trainer.train()

print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
now = datetime.datetime.now()
format_filename = "%d%m_%H%M%S"
file_timestamp = now.strftime(format_filename)

file_name = f"lora{LORA_RANK}_vision_layer_linear_2e-4_bs96_ep2_{file_timestamp}"

model.save_pretrained_merged(file_name, tokenizer,)
