# main.py
import json
import time
import argparse
import os
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
# orchestrator for pipeline variants: omgs, chair_r, chair_e, chair_d, auto
# process_omgs_multi_expert_query: full multi-agent MDT with 5 experts
# process_chair_e_query: CHAIR-E - Single agent evidence-augmented (RAG)
# process_chair_d_query: CHAIR-D - Single agent dossier-augmented (RAG + evidence pack)
# process_chair_r_query: CHAIR-R - Simplest mode records-only (for testing)
# process_auto_query: optional exploratory routing for non-evaluation use
from orchestrator import (
    process_omgs_multi_expert_query,
    process_chair_e_query,
    process_chair_d_query,
    process_chair_r_query,
    process_auto_query,
    normalize_ablation_output_labels,
)
# core for api calls and data loading   
from core import setup_model, load_data, create_question, get_paths_config
# utils for stats collection
from utils.stats_collector import collect_pipeline_stats
from utils.output_bundle import (
    build_manifest,
    extract_case_id,
    next_run_dir,
    sanitize_input_name,
    utc_now,
    write_manifest,
)

# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help="Path to input JSONL file")
parser.add_argument('--model', type=str, default=os.getenv('OMGS_MODEL', 'gpt-5.1'), help="Model/deployment name")
parser.add_argument('--provider', type=str, default=os.getenv('OMGS_PROVIDER', 'auto'),
                    choices=['azure', 'openai', 'openrouter', 'auto'],
                    help="LLM provider: 'azure', 'openai', 'openrouter', or 'auto' (auto-detect based on model name)")
parser.add_argument('--num_samples', type=int, default=999999, help="Number of samples to process")
parser.add_argument('--agent', type=str, default='omgs',
                    choices=['omgs', 'chair_r', 'chair_e', 'chair_d', 'auto'],
                    help="Agent type: 'omgs' (multi-agent), 'chair_r' (CHAIR-R records-only), 'chair_e' (CHAIR-E evidence-augmented), 'chair_d' (CHAIR-D dossier-augmented), 'auto' (optional exploratory routing; not used for reported evaluation)")
parser.add_argument('--omgs_ablation_outputs', action='store_true',
                    help="When agent=omgs, also emit controlled outputs: OMGs w/o Specialist Interpretation, OMGs w/o Deliberation, and full OMGs.")
args = parser.parse_args()

# ---------------------------------------------------------
# 1) Initialize model and client
# ---------------------------------------------------------
model, client = setup_model(args.model, provider=args.provider)
args.client = client
print(f"[INFO] Using provider: {client.provider}, model: {model}")

paths_config = get_paths_config()
db_path = paths_config["output_dirs"]["api_trace_db"]


def _sample_report_path_kwargs(sample):
    report_paths = sample.get("report_paths")
    if not isinstance(report_paths, dict):
        return {}
    key_map = {
        "lab_reports": "labs_json",
        "imaging_reports": "imaging_json",
        "pathology_reports": "pathology_json",
        "mutation_reports": "mutation_json",
    }
    return {
        arg_name: value
        for report_key, arg_name in key_map.items()
        if (value := report_paths.get(report_key))
    }

# ---------------------------------------------------------
# 2) Load data
# ---------------------------------------------------------
test_qa, _ = load_data(
    test_path=args.input_path,
    train_path=None
)

# ---------------------------------------------------------
# 3) Create clean output bundle
# ---------------------------------------------------------
output_root = paths_config["output_dirs"].get("output_answer", "output_answer")
created_at = utc_now()
input_name = sanitize_input_name(args.input_path)
run_id, output_dir = next_run_dir(output_root, input_name, now=created_at)
jsonl_path = output_dir / "results.jsonl"
manifest_path = output_dir / "manifest.json"

write_manifest(
    manifest_path,
    build_manifest(
        run_id=run_id,
        input_path=args.input_path,
        input_count=len(test_qa),
        requested_samples=args.num_samples,
        agent=args.agent,
        model=args.model,
        provider=client.provider,
        output_files=["results.jsonl"],
        created_at=created_at,
        base_dir=Path.cwd(),
    ),
)
jsonl_path.touch()

print(f"[INFO] Output bundle created: {output_dir}")

# ---------------------------------------------------------
# 4) Select agent function
# ---------------------------------------------------------
# Agent selection based on --agent argument
if args.agent == "chair_r":
    process_fn = process_chair_r_query
    print(f"[INFO] Using CHAIR-R (records-only)")
elif args.agent == "chair_e":
    process_fn = process_chair_e_query
    print(f"[INFO] Using CHAIR-E (evidence-augmented)")
elif args.agent == "chair_d":
    process_fn = process_chair_d_query
    print(f"[INFO] Using CHAIR-D (dossier-augmented)")
elif args.agent == "auto":
    process_fn = process_auto_query
    print(f"[INFO] Using Auto - optional exploratory routing (not used for reported evaluation)")
else:
    # Default to OMGs multi-agent
    process_fn = process_omgs_multi_expert_query
    print(f"[INFO] Using OMGs - Multi-agent MDT pipeline")

# ---------------------------------------------------------
# 5) Loop through samples (main loop)
# ---------------------------------------------------------
for no, sample in enumerate(tqdm(test_qa)):
    if no == args.num_samples:
        break

    try:
        question = create_question(sample)

        # Debug: print sample keys for inspection
        # print(f"[DEBUG] Sample keys: {list(sample.keys())}")

        sample_start = datetime.now()
        t0 = time.time()
        args.run_id = sample.get("run_id")
        report_path_kwargs = _sample_report_path_kwargs(sample)

        process_kwargs = dict(
            question=question,
            question_raw=sample.get('question_raw'),
            model=model,
            meta_info=sample.get('meta_info'),
            time=sample.get('Time'),
            args=args,
        )
        if args.agent in {"omgs", "chair_d", "auto"}:
            process_kwargs.update(report_path_kwargs)

        final_decision = process_fn(**process_kwargs)

        sample_elapsed = round(time.time() - t0, 2)
        sample_end = datetime.now()

        # Query token stats from SQLite for this sample's time window
        sample_stats = collect_pipeline_stats(sample_start, sample_end, db_path)
        
        # Determine agent_mode for results (auto mode sets _auto_routed_mode)
        if hasattr(args, '_auto_routed_mode'):
            agent_mode = args._auto_routed_mode
            # Clean up for next sample
            delattr(args, '_auto_routed_mode')
        else:
            agent_mode = args.agent

        summary = dict(sample.get("summary") or {})
        if sample.get("run_id"):
            summary["run_id"] = sample.get("run_id")
        summary["agent_mode"] = agent_mode
        summary["model"] = args.model
        summary["provider"] = args.client.provider if hasattr(args, 'client') else 'unknown'
        summary["inference_time_seconds"] = sample_elapsed
        summary["inference_total_tokens"] = sample_stats.get("total_tokens", 0)
        summary["inference_input_tokens"] = sample_stats.get("total_input_tokens", 0)
        summary["inference_output_tokens"] = sample_stats.get("total_output_tokens", 0)

        result_item = {
            'run_id': run_id,
            'input_name': input_name,
            'input_index': no,
            'case_id': extract_case_id(sample, no),
            'status': 'ok',
            'agent': args.agent,
            'agent_mode': agent_mode,
            'model': args.model,
            'provider': client.provider,
            'response': final_decision,
            'gold_plan': sample.get('gold_plan'),
            'Time': sample.get('Time'),
            'summary': summary,
        }
        ablation_outputs = getattr(args, "_last_ablation_outputs", None)
        if ablation_outputs:
            result_item["ablation_outputs"] = normalize_ablation_output_labels(ablation_outputs)
            delattr(args, "_last_ablation_outputs")

        # Append to JSONL (crash-safe: one line per sample)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

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
        
        # Record error in JSONL
        summary = dict(sample.get("summary") or {})
        summary["agent_mode"] = agent_mode
        summary["model"] = args.model
        summary["provider"] = args.client.provider if hasattr(args, 'client') else 'unknown'
        summary["inference_time_seconds"] = 0
        summary["inference_total_tokens"] = 0
        summary["inference_input_tokens"] = 0
        summary["inference_output_tokens"] = 0
        summary["error"] = str(e)

        result_item = {
            'run_id': run_id,
            'input_name': input_name,
            'input_index': no,
            'case_id': extract_case_id(sample, no),
            'status': 'error',
            'agent': args.agent,
            'agent_mode': agent_mode,
            'model': args.model,
            'provider': client.provider,
            'response': None,
            'gold_plan': sample.get('gold_plan'),
            'Time': sample.get('Time'),
            'error': str(e),
            'summary': summary,
        }

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

    time.sleep(1.1)  # avoid overlapping statistics windows!!!

# ---------------------------------------------------------
# 6) Done
# ---------------------------------------------------------
print(f"[INFO] JSONL saved to {jsonl_path}")
print(f"[INFO] Manifest saved to {manifest_path}")
print(f"[INFO] Done.")
