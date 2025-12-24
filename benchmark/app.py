import gradio as gr
import json
import random
import urllib.parse


# Full dataset
with open('../full_dataset_with_captions_0308.json', 'r') as f:
    full_data = json.load(f)

# Old Captions
with open('../train_dataset_kazvision_26042025.json', 'r') as f1, \
     open('../valid_dataset_kazvision_26042025.json', 'r') as f2:
    old_captions = {**json.load(f1), **json.load(f2)}


def extract_relative_path(img_url: str) -> str:
    """
    Extract relative path from huggingface URL.
    """
    decoded = urllib.parse.unquote(img_url)
    parts = decoded.split('/resolve/main/')
    if len(parts) != 2:
        return ''
    return parts[1].strip().lstrip('/')

def match_old_caption(img_url: str) -> str:
    """
    Return old caption by img_url if matched
    """
    path = extract_relative_path(img_url)
    return old_captions.get(path, None)

# Only for 50 images
matched_items = [item for item in full_data if match_old_caption(item.get('img_url', '')) is not None]

if len(matched_items) >= 50:
    random_samples = random.sample(matched_items, 50)
else:
    unmatched_items = [item for item in full_data if match_old_caption(item.get('img_url', '')) is None]
    random_samples = matched_items + random.sample(unmatched_items, 50 - len(matched_items))

gallery_items = [(item['img_url'], item.get('caption', {}).get('text', 'N/A')) for item in random_samples]

# Gradio
def get_details(evt: gr.SelectData):
    item = random_samples[evt.index]
    if not item:
        return ["", "", "", "", "", ""]

    caption = item.get('caption', {}).get('text', 'N/A')
    old_caption = match_old_caption(item.get('img_url', '')) or 'N/A'

    vqa = item.get('vqa', [])
    vqa_formatted = "\n\n".join([f"Q: {q.get('question', 'N/A')}\nA: {q.get('answer', 'N/A')}" for q in vqa])

    ocr = item.get('ocr', {})
    ocr_formatted = f"Instruction: {ocr.get('instruction', 'N/A')}\nAnswer: {ocr.get('answer', 'N/A')}" if ocr else ""

    reason = item.get('reason', {})
    reason_formatted = f"Instruction: {reason.get('instruction', 'N/A')}\nAnswer: {reason.get('answer', 'N/A')}" if reason else ""

    instruct = item.get('instruct_follow', {})
    instruct_formatted = f"Instruction: {instruct.get('instruction', 'N/A')}\nAnswer: {instruct.get('answer', 'N/A')}" if instruct else ""

    return caption, old_caption, vqa_formatted, ocr_formatted, reason_formatted, instruct_formatted


with gr.Blocks() as demo:
    gr.Markdown("## KazDataset Viewer")
    gr.Markdown("50 images for captions overview.")

    gallery = gr.Gallery(value=gallery_items, label="Изображения", columns=5, height="auto")

    with gr.Row():
        with gr.Column(scale=1):
            caption_output = gr.Textbox(label="New caption")
            old_caption_output = gr.Textbox(label="Old caption")
            vqa_output = gr.Textbox(label="VQA", lines=5)
        with gr.Column(scale=1):
            ocr_output = gr.Textbox(label="OCR")
            reason_output = gr.Textbox(label="Reasoning")
            instruct_follow_output = gr.Textbox(label="Instruct Follow")

    gallery.select(
        fn=get_details,
        outputs=[
            caption_output,
            old_caption_output,
            vqa_output,
            ocr_output,
            reason_output,
            instruct_follow_output
        ]
    )

if __name__ == "__main__":
    demo.launch()
