# SPDX-License-Identifier: Apache-2.0
import os
from typing import NamedTuple, Optional, List, Tuple, Dict
import json
from dataclasses import asdict
from typing import NamedTuple, Optional, List, Tuple
from contextlib import contextmanager
from tqdm import tqdm
import PIL
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

# --------- 模型别名路由保持不变 ----------
def run_qwen2_5_vl(input_model_name, questions, modality) -> ModelRequestData:
    local_aliases = {
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    }
    
    model_name = local_aliases.get(input_model_name, input_model_name)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=10,
        mm_processor_kwargs={
            'min_pixels': 256 * 28 * 28,
            'max_pixels': 1280 * 28 * 28,
        },
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    return ModelRequestData(engine_args=engine_args, prompts=[""])


def build_question_text(item: Dict) -> str:

    return (
        "Based on these conditions, answer the question: "
        + item["text"]
        + "You should FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        + "The reasoning process MUST BE enclosed within <think> </think> tags. "
        + "The final answer MUST BE put in \\boxed{}"
    )

def render_prompt_and_stops(
    model_name: str,
    question: str,
) -> Tuple[str, Optional[List[int]], AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
    stop_tokens = None
    
    if stop_tokens:
        stop_ids = [tok.convert_tokens_to_ids(t) for t in stop_tokens]

    return prompt, stop_ids

@contextmanager
def time_counter(enable: bool):
    if enable:
        import time
        start = time.time()
        yield
        print("-"*50)
        print("-- generate time = {:.3f}s".format(time.time()-start))
        print("-"*50)
    else:
        yield

def parse_args():
    parser = FlexibleArgumentParser(description='vLLM offline inference (VLM)')
    parser.add_argument('--num-prompts', type=int, default=4, help='Samples per question (n).')
    parser.add_argument('--modality', type=str, default="image", choices=['image','video'])
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.6)  
    parser.add_argument('--image-repeat-prob', type=float, default=None)  # 保留但不使用
    parser.add_argument('--disable-mm-preprocessor-cache', action='store_true')
    parser.add_argument('--time-generate', action='store_true')
    parser.add_argument('--use-different-prompt-per-request', action='store_true')  # 保留但不使用
    parser.add_argument('--ip-model-name', required=True)
    parser.add_argument('--max-tokens', type=int, default=4096)  # 新增：更合理的默认
    return parser.parse_args()

def main(args):
    model = args.ip_model_name
    modality = args.modality

    with open("PGPS9K/test.json", "r") as f:
        test_items = json.load(f)
    question_keys = sorted(test_items.keys())
    import random
    random.seed(args.seed)
    
    req = run_qwen2_5_vl(model, [""], modality)
    resolved_model_name = req.engine_args.model  
    engine_args = asdict(req.engine_args) | {
        "seed": args.seed,
        "disable_mm_preprocessor_cache": args.disable_mm_preprocessor_cache,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,  
    }
    llm = LLM(**engine_args)

    inputs = []
    global_stop_ids: Optional[List[int]] = None
    for qk in tqdm(question_keys):
        qtext = build_question_text(test_items[qk])
        img_path = os.path.join("Diagram_Visual", test_items[qk]['diagram'])
        data = PIL.Image.open(img_path).convert("RGB")

        prompt, stop_ids = render_prompt_and_stops(resolved_model_name, qtext)
        if stop_ids and global_stop_ids is None:
            global_stop_ids = stop_ids

        inputs.append({"prompt": prompt, "multi_modal_data": {modality: data}})

    sampling_params = SamplingParams(
        temperature=args.temperature,
        n=args.num_prompts,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    with time_counter(args.time_generate):
        outputs = llm.generate(inputs, sampling_params=sampling_params,
                               lora_request=req.lora_requests)

    out_dir = f"results/new_9k_{model}_{args.max_tokens}_tmp_{args.temperature}"
    os.makedirs(out_dir, exist_ok=True)
    for qi, o in enumerate(outputs):
        qid = question_keys[qi]
        for j, cand in enumerate(o.outputs):
            with open(os.path.join(out_dir, f"q{qid}_attempt{j}.txt"), "w") as f:
                f.write(cand.text)

if __name__ == "__main__":
    args = parse_args()
    main(args)
