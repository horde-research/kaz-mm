import json
import os

# Define task names
tasks = ['caption', 'reason', 'ocr', 'vqa', 'inst_follow']

# Define file paths
evaluate_dir = 'RL_results'
finetune_dir = 'SFT_preds'

def merge_json_files(task_name):
    """
    Merge corresponding JSON files from evaluate and finetune directories.
    
    Args:
        task_name: Name of the task (e.g., 'caption', 'reason', etc.)
    """
    # Construct file paths
    eval2_path = os.path.join(evaluate_dir, f'{task_name}_task_eval.json')
    eval_path = os.path.join(finetune_dir, f'{task_name}_task_eval.json')
    output_path = os.path.join(evaluate_dir, f'{task_name}_task_eval_merged.json')
    
    try:
        # Read the first file (from evaluate directory)
        with open(eval2_path, 'r') as f:
            eval2_data = json.load(f)
        
        # Read the second file (from finetune directory)
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)
        
        # Merge the data
        merged_data = {}
        
        # Get all unique keys from both files
        all_keys = set(eval2_data.keys()) | set(eval_data.keys())
        
        for key in all_keys:
            merged_data[key] = {}
            
            # Add data from eval2 file
            if key in eval2_data:
                merged_data[key].update(eval2_data[key])
            
            # Add data from eval file
            if key in eval_data:
                merged_data[key].update(eval_data[key])
        
        # Write merged data to output file
        os.makedirs(finetune_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"✓ Successfully merged {task_name}_task files -> {output_path}")
        
    except FileNotFoundError as e:
        print(f"✗ Error processing {task_name}_task: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"✗ Error processing {task_name}_task: Invalid JSON - {e}")
    except Exception as e:
        print(f"✗ Error processing {task_name}_task: {e}")

def main():
    """
    Main function to merge all task files.
    """
    print("Starting JSON file merge process...\n")
    
    for task in tasks:
        merge_json_files(task)
    
    print("\nMerge process completed!")

if __name__ == "__main__":
    main()