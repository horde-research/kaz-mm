## Model benchmarking

To serve a model:

## QWEN2.5 VL

`vllm serve Qwen/Qwen2.5-VL-7B-Instruct   --port 8000   --host 0.0.0.0   --mm-processor-kwargs '{"max_pixels": 1638400}'`


## INTERNVL3

`lmdeploy serve api_server OpenGVLab/InternVL3-8B --chat-template internvl2_5 --server-port 23333 --tp 1`

## Gemma 4b

`vllm serve google/gemma-3-4b-it --port 8000 --host 0.0.0.0`


## LLama 3.2 Vision

`vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct --port 8000 --host 0.0.0.0`

