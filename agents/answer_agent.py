#!/usr/bin/python3

import re
import json

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent
# from .answer_model_llama import AAgent

class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""

    def __init__(self, **kwargs):
        self.agent = AAgent(**kwargs)

    def build_prompt(self, question_data: Dict[str, str | Any]) -> Tuple[str, str]:
        """Generate answer with reasoning for AAIPL competition MCQs"""

        # System prompt focused on the 4 AAIPL competition topics
        sys_prompt = (
            "You are an elite logical reasoning and puzzle-solving agent specializing in "
            "competitive exam MCQs. Your expertise covers:\n"
            "- Syllogisms (premises, conclusions, deductive reasoning)\n"
            "- Seating Arrangements (linear and circular positioning logic)\n"
            "- Blood Relations & Family Tree (relationship inference)\n"
            "- Mixed Series (alphanumeric pattern recognition)\n\n"
            "CRITICAL RULES:\n"
            "- Think step-by-step before answering\n"
            "- Analyze each option methodically; eliminate distractors\n"
            "- For syllogisms: draw Venn diagrams mentally, check all premises\n"
            "- For seating: track positions carefully, note directions (left/right, clockwise)\n"
            "- For blood relations: map the family tree step by step\n"
            "- For series: identify separate patterns for letters and numbers\n"
            "- If multiple answers seem valid, choose the MOST ACCURATE one\n"
            "- Always output valid JSON - no extra text before or after"
        )

        # User prompt with domain-adaptive instructions
        user_prompt = (
            "Answer the following multiple-choice question:\n\n"
            "QUESTION: {question}\n\n"
            "OPTIONS:\n{options}\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify the topic (syllogisms/seating/blood-relations/series)\n"
            "2. Apply step-by-step logical reasoning specific to that topic\n"
            "3. Analyze each option carefully - watch for traps and distractors\n"
            "4. Eliminate wrong options systematically\n"
            "5. Select the single best answer\n\n"
            "OUTPUT FORMAT (strict JSON only):\n"
            "{{\n"
            '  "answer": "C",\n'
            '  "reasoning": "Step-by-step analysis within 100 words"\n'
            "}}\n\n"
            "NOW SOLVE THIS QUESTION:"
        )

        # Format the actual question
        question_text = question_data.get('question', '')
        choices = question_data.get('choices', [])

        # Format options using existing helper
        options_formatted = self._format_choices(choices)

        final_user_prompt = user_prompt.format(
            question=question_text,
            options=options_formatted
        )

        return final_user_prompt, sys_prompt

    def answer_question(
        self, question_data: Dict | List[Dict], **kwargs
    ) -> Tuple[List[Dict], int | None, float | None]:
        """Generate answer(s) for the given question(s)"""
        if isinstance(question_data, list):
            prompt = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(question_data)

        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if (
            isinstance(resp, list) and all(isinstance(r, str) for r in resp)
        ) or isinstance(resp, str):
            return resp, tl, gt
        else:
            return (
                "",
                tl,
                gt if not isinstance(resp, list) else [""] * len(resp),
                tl,
                gt,
            )

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            batch_answers, tl, gt = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        # Handle last batch with less than batch_size
        if len(questions) % batch_size != 0:
            batch_questions = questions[-(len(questions) % batch_size) :]
            batch_answers = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers[0])
            tls.append(batch_answers[1])
            gts.append(batch_answers[2])
            pbar.update(1)
        pbar.close()
        return answers, tls, gts

    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError("The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(self, ans: List[str | Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format"""

        def basic_checks(a1: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["answer"]
            if all((key in a1) and isinstance(a1[key], str) for key in required_keys):
                if len(a1["answer"]) == 1 and (a1["answer"] not in "ABCDabcd"):
                    return False
                check_len = self.count_tokens_a(a1["answer"])
                if check_len < 50:
                    check_len += self.count_tokens_a(a1.get("reasoning", "None"))
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
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
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

    def save_answers(self, answers: List[str], file_path: str | Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([a for a in answers], f, indent=4)

    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            # Ensure each choice starts with a letter if not already formatted
            if not re.match(r"^[A-D]\)", choice.strip()):
                # Extract letter from existing format or assign based on position
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return "\n".join(formatted)


# Example usage
if __name__ == "__main__":
    import json
    import yaml
    import argparse
    from utils.build_prompt import auto_json, option_extractor_prompt

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # python -m agents.answer_agent --input_file outputs/filtered_questions.json --output_file outputs/answers.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    argparser = argparse.ArgumentParser(description="Run the Answering Agent")
    argparser.add_argument(
        "--input_file",
        type=str,
        default="outputs/filtered_questions.json",
        help="Path to the input JSON file with questions",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/answers.json",
        help="Path to save the answers",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for processing questions"
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = argparser.parse_args()

    # Load sample questions (assuming they're saved from QuestioningAgent)
    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent()

    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 512, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))
    answer, tls, gts = agent.answer_batches(
        questions=sample_questions, batch_size=args.batch_size, **gen_kwargs
    )
    ans = []
    for idx, (q, a) in enumerate(zip(sample_questions, answer)):
        if args.verbose:
            print(f"\n=== Question {idx+1} ===")
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Model Answer:\n{a}")
        try:
            a = json.loads(a)
            if all(k in a for k in ["answer", "reasoning"]):
                # ++++++++++++++++++++++++++
                # TODO: IMPROVE THE FOLLOWING
                if len(a["answer"]) != 1:
                    a["answer"] = agent.agent.generate_response(
                        option_extractor_prompt(a["answer"], q["choices"])
                    )
                # ++++++++++++++++++++++++++
            else:
                # the dictionary is not as expected. So extract it using the same model: Self-Reflection
                prompt = (
                    "Extract **ONLY** the answer and reasoning while discarding the rest.\n\n"
                    "String:\n"
                    "{}\n\n"
                    "Given Format:\n"
                    "{{\n"
                    '    "answer": "Only the option letter (A, B, C, or D)",\n'
                    '    "reasoning": "..."\n'
                    "}}"
                )
                a = agent.agent.generate_response(
                    prompt.format(json.dumps(a, indent=4))
                )
        except json.JSONDecodeError:
            a = agent.agent.generate_response(auto_json(a))
        ans.append(a)

    if args.verbose:
        if gen_kwargs.get("tgps_show", False):
            for idx, (tl, gt) in enumerate(zip(tls, gts)):
                print(f"BATCH - {idx}")
                print(f"Tokens: {tl}, Time: {gt:.3f} seconds")
                print(f"TGPS: {tl/gt:.3f} seconds")
            print("\n" + "=" * 50)
            print(
                f"Total Time: {sum(gts):.3f} seconds; Total Tokens: {sum(tls)}; TGPS: {sum(tls)/sum(gts):.3f} seconds"
            )

    # Save answers
    agent.save_answers(ans, args.output_file)
    filtered_file_name = args.output_file.replace(
        "answers.json", "filtered_answers.json"
    )
    agent.save_answers(agent.filter_answers(ans), filtered_file_name)
