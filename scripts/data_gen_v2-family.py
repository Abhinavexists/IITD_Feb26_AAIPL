import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import pandas as pd

client = AsyncOpenAI(base_url='http://0.0.0.0:8000/v1',api_key='nvm')
model_name="gptoss120b/"
async def generate_async(prompt, idx):
    try:
        response = await client.responses.create(
            model=model_name,  # or your preferred model
            input=[
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0
        )
        
        return idx, response.output[1].content[0].text
    except Exception as e:
        return idx, f"Error: {str(e)}"

developer_prompt="""You are an expert-level examiner creating extremely difficult MCQ questions about {topic}.

CRITICAL RULES:
1. Generate ONLY ONE question per response (not multiple)
2. Topic must be strictly: {topic}
3. Make questions genuinely hard - trick 50%+ of experts
4. Return ONLY valid JSON (no other text)

FORMAT (must be exact):
{{
    "topic": "<Topic of the Question>",
    "question": "<full question text>",
    "choices": [
        "A) <choice A text>",
        "B) <choice B text>",
        "C) <choice C text>",
        "D) <choice D text>"
    ],
    "answer": "<correct choice letter only>",
    "explanation": "brief explanation within 100 words for why the answer is correct"
}}"""

async def main():
    topics = [
    "Logical Reasoning: Syllogisms",
    "Puzzles: Seating Arrangements (Linear, Circular)",
    "Mixed Series (Alphanumeric)",
    "Blood Relations and Family Tree: Family Tree logic"]

    tasks = [generate_async(developer_prompt.format(topic=topics[3]), idx) for idx in range(1000)]

    results = [None] * len(tasks)

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        idx, output = await future
        results[idx] = output

    return results


#results = await main()

results = asyncio.run(main())

final_output=[]
for i, r in enumerate(results):
    print(f"\n[{i}] {r}")
    final_output.append(eval(r))
    pd.DataFrame({'answer':final_output}).to_csv('Family_Tree.csv',index=False)
