# main.py
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime

from host import process_omgs_multi_expert_query
from core import setup_model, load_data, create_question

# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help="Path to input JSONL file")
parser.add_argument('--model', type=str, default='gpt-5.1', help="Azure Deployment Name")
parser.add_argument('--num_samples', type=int, default=999999, help="Number of samples to process")
parser.add_argument('--agent', type=str, default='basic_baseline',
                    choices=['basic_baseline', 'basic_role','basic_rag','basic_rag_lab','basic_rag_lab_full','omgs'],
                    help="Choose which reasoning agent to use")
args = parser.parse_args()

# ---------------------------------------------------------
# 1) Initialize model and client
# ---------------------------------------------------------
model, client = setup_model(args.model)
args.client = client

# ---------------------------------------------------------
# 2) Load data
# ---------------------------------------------------------
test_qa, _ = load_data(
    test_path=args.input_path,
    train_path=None
)

# ---------------------------------------------------------
# 3) Create timestamped output folder
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"output_answer/{args.agent}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

txt_path = f"{output_dir}/results.txt"
json_path = f"{output_dir}/results.json"

# Reset TXT
open(txt_path, "w").close()

print(f"[INFO] Output folder created: {output_dir}")

# ---------------------------------------------------------
# 4) Select agent function
# ---------------------------------------------------------
# Currently only OMGs agent is supported
if args.agent != "omgs":
    print(f"[WARNING] Agent '{args.agent}' not implemented, using 'omgs' instead")
process_fn = process_omgs_multi_expert_query


results = []

# ---------------------------------------------------------
# 5) Loop through samples
# ---------------------------------------------------------
for no, sample in enumerate(tqdm(test_qa)):
    if no == args.num_samples:
        break

    try:
        question = create_question(sample)

        # Debug: print sample keys for inspection
        print(f"[DEBUG] Sample keys: {list(sample.keys())}")
        final_decision = process_fn(
            question=question,
            question_raw = sample.get('question_raw'),
            model=model,
            meta_info=sample.get('meta_info'),  
            time = sample.get('Time'),
            args=args
        )

        results.append({
            'scene': sample.get('scene'),
            'question': question,
            'response': final_decision,
            'gold_plan': sample.get('gold_plan'),
            'question_raw': sample.get('question_raw'),
            'Time': sample.get('Time'),
            'meta_info': sample.get('meta_info'),

        })

        # Write to TXT
        with open(txt_path, "a", encoding="utf-8") as ftxt:
            ftxt.write("====================\n")
            ftxt.write("QUESTION:\n")

            # `question` may be dict/list/str; normalize to stable text before writing
            if isinstance(question, (dict, list)):
                question_text = json.dumps(question, ensure_ascii=False, indent=2)
            else:
                question_text = str(question).strip()

            ftxt.write(question_text + "\n\n")
            ftxt.write("RESPONSE:\n")
            ftxt.write(str(final_decision).strip() + "\n\n")
    
    except Exception as e:
        # Log error but continue processing other samples
        error_msg = f"Error processing sample {no}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # Record error in results
        results.append({
            'scene': sample.get('scene'),
            'question': sample.get('question', ''),
            'response': None,
            'error': str(e),
            'gold_plan': sample.get('gold_plan'),
            'question_raw': sample.get('question_raw'),
            'Time': sample.get('Time'),
            'meta_info': sample.get('meta_info'),
        })
        
        # Write error to TXT file
        with open(txt_path, "a", encoding="utf-8") as ftxt:
            ftxt.write("====================\n")
            ftxt.write("QUESTION:\n")
            ftxt.write(json.dumps(sample, ensure_ascii=False, indent=2) + "\n\n")
            ftxt.write("ERROR:\n")
            ftxt.write(error_msg + "\n\n")

# ---------------------------------------------------------
# 6) Save JSON
# ---------------------------------------------------------
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"[INFO] TXT saved to {txt_path}")
print(f"[INFO] JSON saved to {json_path}")
print(f"[INFO] Done.")
