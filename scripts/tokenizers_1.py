# import os
# os.environ["HF_HOME"] = "/tmp/hf_cache"
# os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import os, requests
from typing import List, Dict

os.makedirs("/tmp/qwen_tokenizer", exist_ok=True)

base_url = "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/"
files = ["tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.json"]

for f in files:
    path = f"/tmp/qwen_tokenizer/{f}"
    if not os.path.exists(path):
        print(f"Downloading {f}...")
        r = requests.get(base_url + f, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as fp:
            fp.write(r.content)
        print(f"  Done ({len(r.content)} bytes)")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/tmp/qwen_tokenizer", padding_side='left')
print("Tokenizer loaded!")

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", padding_side='left')

def count_tokens_a(text: str) -> int:
    """Count the number of tokens in the text using the agent's tokenizer"""
    return len(tokenizer.encode(text, add_special_tokens=False))

def filter_answers(ans: List[str|Dict[str, str]]) -> List[Dict[str, str]]:
    r"""Filter answers to ensure they are in the correct format"""
    def basic_checks(a1: Dict[str, str])->bool:
        # check required keys
        required_keys = ['answer']
        if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
            if len(a1['answer']) == 1 and (a1['answer'] not in 'ABCDabcd'):
                    return False
            check_len = count_tokens_a(a1['answer'])
            if check_len < 50:
                check_len += count_tokens_a(a1.get('reasoning', 'None'))
                if check_len < 512:
                    # check answer format - EXTRA checks
                    # if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                    return True
        return False

    filtered_answers = []
    for i, a in enumerate(ans):
        if isinstance(a, dict):
            if basic_checks(a):
                filtered_answers.append(a)
            else:
                filtered_answers.append(None)
        elif isinstance(a, str):
            # Basic checks: at least with correct JSON format
            try:
                a1 = json.loads(a)
                if basic_checks(a1):
                    filtered_answers.append(a1)
                else:
                    filtered_answers.append(None)
            except json.JSONDecodeError:
                # If JSON decoding fails, skip this answer
                print(f"Skipping invalid JSON at index {i}: {a}")
                filtered_answers.append(None)
                continue
        else:
            # If the answer is neither a dict nor a str, skip it
            print(f"Skipping unsupported type at index {i}: {type(a)}")
            filtered_answers.append(None)
    return filtered_answers