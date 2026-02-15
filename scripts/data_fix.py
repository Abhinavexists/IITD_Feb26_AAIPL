import json
import ast

# Load the malformed data
print("Loading questions_training.json...")
with open('questions_training.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} items")

# Fix the structure
fixed_data = []
errors = 0

for idx, item in enumerate(data):
    try:
        # Parse the stringified dict in the "answer" field
        if isinstance(item.get('answer'), str):
            # Try to parse as Python literal
            question_data = ast.literal_eval(item['answer'])
        else:
            question_data = item['answer']

        fixed_data.append(question_data)
    except Exception as e:
        errors += 1
        if errors <= 3:  # Print first 3 errors
            print(f"Error at index {idx}: {e}")
        continue

# Save the fixed data
print(f"\nFixed {len(fixed_data)} questions ({errors} errors)")
with open('questions_training_fixed.json', 'w') as f:
    json.dump(fixed_data, f)

print("Saved to questions_training_fixed.json")

# Verify the first item
if fixed_data:
    print("\nFirst fixed item:")
    print(json.dumps(fixed_data[0], indent=2))
