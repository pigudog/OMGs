# main.py
import os
import json
import argparse
from tqdm import tqdm
from datetime import datetime

from host import (
    process_omgs_multi_expert_query,
    process_chair_sa_k_query,
    process_chair_sa_kep_query,
    process_chair_sa_query,
    process_auto_query,
)
from core import setup_model, load_data, create_question

# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help="Path to input JSONL file")
parser.add_argument('--model', type=str, default='gpt-5.1', help="Model/deployment name")
parser.add_argument('--provider', type=str, default='auto',
                    choices=['azure', 'openai', 'openrouter', 'auto'],
                    help="LLM provider: 'azure', 'openai', 'openrouter', or 'auto' (auto-detect based on model name)")
parser.add_argument('--num_samples', type=int, default=999999, help="Number of samples to process")
parser.add_argument('--agent', type=str, default='omgs',
                    choices=['omgs', 'chair_sa', 'chair_sa_k', 'chair_sa_kep', 'auto'],
                    help="Agent type: 'omgs' (multi-agent), 'chair_sa' (simplest), 'chair_sa_k' (knowledge), 'chair_sa_kep' (knowledge+evidence), 'auto' (intelligent routing)")
args = parser.parse_args()

# ---------------------------------------------------------
# 1) Initialize model and client
# ---------------------------------------------------------
model, client = setup_model(args.model, provider=args.provider)
args.client = client
print(f"[INFO] Using provider: {client.provider}, model: {model}")

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
# Agent selection based on --agent argument
if args.agent == "chair_sa":
    process_fn = process_chair_sa_query
    print(f"[INFO] Using Chair-SA - Simplest mode (for testing)")
elif args.agent == "chair_sa_k":
    process_fn = process_chair_sa_k_query
    print(f"[INFO] Using Chair-SA(K) - Single agent with Knowledge only")
elif args.agent == "chair_sa_kep":
    process_fn = process_chair_sa_kep_query
    print(f"[INFO] Using Chair-SA(K+EP) - Single agent with Knowledge + Evidence Pack")
elif args.agent == "auto":
    process_fn = process_auto_query
    print(f"[INFO] Using Auto - Intelligent routing based on case complexity")
else:
    # Default to OMGs multi-agent
    process_fn = process_omgs_multi_expert_query
    print(f"[INFO] Using OMGs - Multi-agent MDT pipeline")


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
        
        # Determine agent_mode for results (auto mode sets _auto_routed_mode)
        if hasattr(args, '_auto_routed_mode'):
            agent_mode = args._auto_routed_mode
            # Clean up for next sample
            delattr(args, '_auto_routed_mode')
        else:
            agent_mode = args.agent

        results.append({
            'agent_mode': agent_mode,
            'mode': agent_mode,
            'model': args.model,
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
        
        # Determine agent_mode for error case
        if hasattr(args, '_auto_routed_mode'):
            agent_mode = args._auto_routed_mode
            delattr(args, '_auto_routed_mode')
        else:
            agent_mode = args.agent
        
        # Record error in results
        results.append({
            'agent_mode': agent_mode,
            'mode': agent_mode,
            'model': args.model,
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
