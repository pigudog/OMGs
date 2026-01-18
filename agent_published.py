# agent_published.py
# NOTE: This file is part of the published OMGs/MDT agent pipeline.
# Any changes here should preserve clinical logic and only adjust observability/debugging unless explicitly intended.
import os
import re
import sys
import json
import torch
import html as _html
from typing import Any, Dict, List, Optional
from chromadb import PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from utils.console_utils import Color, normalize_trial_compact,safe_parse_json_block,question_to_text,generate_final_output,assistant_trial_suggestion
from utils.omgs_reports import save_mdt_log,save_case_html_report
from utils import make_cutoff,parse_dt,safe_date10,filter_before,report_range
from utils import build_lab_timeline,build_imaging_timeline,build_pathology_timeline
from utils import VisualConfig, TraceLogger,print_selected_reports_table,print_section,print_rag_hits_table
from aoai import OpenAIWrapper
from utils import load_patient_labs,load_patient_imaging,load_patient_pathology,load_patient_mutations,read_jsonl,parse_ids,parse_date_any,summarize_selected_reports
from utils.rag_utils import rag_search_pack,build_rag_query_for_mdt,summarize_rag_evidence
from utils.role_utils import ROLES,ROLE_PERMISSIONS,ROLE_PROMPTS,safe_load_case_json,expert_select_reports,init_expert_agent
from utils.core import init_client,Agent,load_data,create_question,setup_model,load_paths_config,get_paths_config
# Public API of `utils`
import random
import hashlib
import tiktoken
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from prettytable import PrettyTable, ALL
from dataclasses import dataclass, field
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################
# 6. MULTI-ROUND MDT DISCUSSION ENGINE
###############################################################################
def run_mdt_discussion(
    agents,
    assistant,
    num_rounds=2,
    num_turns=2,
    max_merged_chars=5000,
    max_turn_delta_chars=900,
    max_targets_per_speaker=4,
    visit_time: Optional[str] = None,
):
    """
    Run multi-round MDT discussion engine with role-based agents.
    
    Note: Each agent's system prompt already contains case_view + clinical reports + 
    global_guideline_digest, so we don't need to repeat the full question in each round.
    
    Args:
        agents: Dictionary mapping role names to Agent instances
        assistant: Assistant agent for summarizing discussions
        num_rounds: Number of discussion rounds
        num_turns: Number of turns per round
        max_merged_chars: Maximum characters to keep in merged context
        max_turn_delta_chars: Maximum characters per turn delta
        max_targets_per_speaker: Maximum targets a speaker can address per turn
        visit_time: Optional visit timestamp for temporal context
    
    Returns:
        Tuple of (initial_ops, merged_context, final_round_ops, interaction_log)
    """

    def _clip_tail(s: str, max_chars: int) -> str:
        if not s:
            return ""
        return s if len(s) <= max_chars else s[-max_chars:]

    def _append_bounded(base: str, addition: str, max_chars: int) -> str:
        if not addition:
            return _clip_tail(base, max_chars)
        if not base:
            return _clip_tail(addition, max_chars)
        return _clip_tail(base + "\n" + addition, max_chars)

    print(f"{Color.OKCYAN}{Color.BOLD}🧠 Starting MDT Discussion Engine...{Color.RESET}")
    agent_list = list(agents.keys())

    emoji_pool = ["👨‍⚕️","👩‍⚕️","🧑‍⚕️","👨🏻‍⚕️","👩🏼‍⚕️","👨🏽‍⚕️","👩🏽‍⚕️","👨🏾‍⚕️","👩🏾‍⚕️","🧑🏾‍⚕️"]
    random.shuffle(emoji_pool)
    role_to_emoji = {r: emoji_pool[i % len(emoji_pool)] for i, r in enumerate(agent_list)}
    chair_role = "chair" if "chair" in agent_list else agent_list[0]

    last_msg_by_pair = {}

    # INITIAL OPINIONS
    print(f"{Color.BOLD}{Color.OKBLUE}\n📌 Collecting Initial Opinions...{Color.RESET}")
    initial_ops = {}
    for role, ag in agents.items():
        print(f"{Color.OKGREEN} - {role}:{Color.RESET}")
        op = ag.chat(
            "Give INITIAL opinion (use ONLY your system-provided patient facts).\n"
            "Return up to 5 bullets, each ≤20 words.\n"
            "If key data missing, say exactly what needs updating."
        )
        print(f"{Color.OKCYAN}   {role_to_emoji[role]} {op}{Color.RESET}")
        initial_ops[role] = op

    merged = assistant.chat(
        "Summarize expert opinions concisely for MDT.\n"
        f"{json.dumps(initial_ops, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Output:\nKey Knowledge:\n- ...\nControversies:\n- ...\nMissing Info:\n- ...\nWorking Plan:\n- ..."
    )
    merged = _clip_tail(merged, max_merged_chars)

    interaction_log = {
        f"Round {r}": {
            f"Turn {t}": {s: {d: None for d in agent_list} for s in agent_list}
            for t in range(1, num_turns + 1)
        }
        for r in range(1, num_rounds + 1)
    }
    final_round_ops = {}

    # MAIN DISCUSSION
    for r in range(1, num_rounds + 1):
        print(f"{Color.WARNING}{Color.BOLD}\n==================== ROUND {r} ===================={Color.RESET}")
        round_key = f"Round {r}"

        summary = assistant.chat(
            f"MDT global knowledge:\n{merged}\n\n"
            "Re-summarize concisely. Must include:\n"
            "Key Knowledge:\n- ...\nControversies:\n- ...\nMissing Info:\n- ...\nWorking Plan:\n- ..."
        )
        merged = _clip_tail(summary, max_merged_chars)

        MDT_should_stop = False

        for t in range(1, num_turns + 1):
            print(f"{Color.BOLD}{Color.OKCYAN}\n--- Turn {t} ---{Color.RESET}")
            turn_key = f"Turn {t}"
            num_speakers = 0
            turn_msgs_compact = []
            ctx_for_turn = merged

            for role, ag in agents.items():
                allowed_targets = [x for x in agent_list if x != role]
                allowed_targets_str = ",".join(allowed_targets)

                speak_prompt = (
                    f"ROLE: {role}. VISIT: {visit_time or 'Unknown'}\n"
                    "Default is NOT to speak. Speak ONLY if: conflict | safety | missing-critical | new-critical.\n\n"
                    f"CONTEXT (latest):\n{ctx_for_turn}\n\n"
                    f"Allowed targets: [{allowed_targets_str}]\n"
                    "Return ONE-LINE JSON only:"
                    '{"speak":"yes/no","messages":[{"target":"<role>","message":"<1-2 sentences>","why":"conflict|safety|missing|new"}]}'
                )

                resp = ag.chat(speak_prompt)
                data = safe_parse_json_block(resp)

                if not data or str(data.get("speak", "no")).lower() != "yes":
                    continue

                msgs = data.get("messages", None)
                if not isinstance(msgs, list):
                    old_msg = (data.get("message") or "").strip()
                    old_targets = data.get("targets") or []
                    if old_msg and isinstance(old_targets, list) and old_targets:
                        msgs = [{"target": tr, "message": old_msg, "why": "unspecified"} for tr in old_targets]
                    else:
                        continue

                accepted_any = False
                used_targets = set()

                for item in msgs:
                    if not isinstance(item, dict):
                        continue
                    target = item.get("target", None)
                    msg = (item.get("message") or "").strip()
                    why = (item.get("why") or "").strip().lower()

                    if not msg:
                        continue
                    if why not in {"conflict", "safety", "missing", "new"}:
                        why = "unspecified"

                    if target not in allowed_targets:
                        target = chair_role
                    if target == role:
                        continue

                    if target in used_targets:
                        continue
                    if len(used_targets) >= max_targets_per_speaker:
                        break

                    key = (role, target)
                    if last_msg_by_pair.get(key) == msg:
                        continue
                    last_msg_by_pair[key] = msg

                    used_targets.add(target)
                    accepted_any = True

                    interaction_log[round_key][turn_key][role][target] = msg
                    print(f"{Color.OKGREEN}  {role_to_emoji[role]} {role} → {role_to_emoji[target]} {target}:{Color.RESET} [{why}] {msg}")

                    turn_msgs_compact.append(f"{role}->{target}({why}): {msg}")

                if accepted_any:
                    num_speakers += 1

            if num_speakers == 0:
                print(f"{Color.WARNING} ⚠ No experts spoke in this turn → Skip remaining turns and finalize this round.{Color.RESET}")
                MDT_should_stop = True
                break

            if turn_msgs_compact:
                delta_text = " | ".join(turn_msgs_compact)
                delta_text = _clip_tail(delta_text, max_turn_delta_chars)
                merged = _append_bounded(merged, f"[R{r}T{t} DELTA] {delta_text}", max_merged_chars)

        # FINAL plans for this round
        final_round_ops[round_key] = {}
        print(f"{Color.BOLD}{Color.OKBLUE}\n📘 Finalizing Expert Plans for ROUND {r} ...{Color.RESET}")
        for role, ag in agents.items():
            print(f"{Color.OKBLUE}{Color.BOLD} - {role}:{Color.RESET}")
            final_op = ag.chat(
                f"Given MDT context:\n{merged}\n\n"
                "Provide FINAL refined plan.\n"
                "Up to 5 bullets, each ≤20 words.\n"
                "Any factual claim must include [@report_id|date] or say unknown."
            )
            print(f"{Color.OKGREEN}{final_op}{Color.RESET}\n")
            final_round_ops[round_key][role] = final_op

        if MDT_should_stop:
            print(f"{Color.WARNING}{Color.BOLD}🚫 MDT stopped early after Round {r}. No further rounds will be executed.{Color.RESET}")
            return initial_ops, merged, final_round_ops, interaction_log

    return initial_ops, merged, final_round_ops, interaction_log


###############################################################################
# 7. INTERACTION DIRECTION MATRIX（PrettyTable）
###############################################################################
def _count_interactions(interaction_log, src, dst):
    c = 0
    for rnd in interaction_log.values():
        for turn in rnd.values():
            msg = turn.get(src, {}).get(dst)
            if msg:
                c += 1
    return c


def print_interaction_matrix(interaction_log, roles_order=ROLES):
    print(f"\n{Color.BOLD}{Color.OKBLUE}📊 Interaction Direction Matrix (All Rounds × Turns){Color.RESET}")
    print("Legend: . none | ->N A→B count | <-N B→A count | <->a/b both directions\n")

    agent_list = list(roles_order)
    tbl = PrettyTable([""] + agent_list)
    tbl.align = "c"
    tbl.hrules = ALL
    tbl.vrules = ALL
    tbl.padding_width = 1

    for A in agent_list:
        row = [A]
        for B in agent_list:
            if A == B:
                row.append("")
                continue
            a2b = _count_interactions(interaction_log, A, B)
            b2a = _count_interactions(interaction_log, B, A)
            if a2b == 0 and b2a == 0:
                cell = "."
            elif a2b > 0 and b2a == 0:
                cell = f"->{a2b}"
            elif a2b == 0 and b2a > 0:
                cell = f"<-{b2a}"
            else:
                cell = f"<->{a2b}/{b2a}"
            row.append(cell)
        tbl.add_row(row)
    print(tbl)



###############################################################################
# 9. MAIN ENTRY
###############################################################################
def process_omgs_multi_expert_query(
    question,
    question_raw,
    model,
    args,
    time=None,
    meta_info=None,
    labs_json=None,
    imaging_json=None,
    pathology_json=None,
    mutation_json=None,
    device="auto",
    topk=5,
    case_filter_buffer_days=120,
    strict_context_prune=False,
    trials_json_path=None
):
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== MDT Multi-Expert Pipeline Start ==={Color.RESET}")
    
    # Load paths configuration
    paths_config = get_paths_config()
    
    # Use config paths if not explicitly provided (backward compatibility)
    if labs_json is None:
        labs_json = paths_config["data_files"]["lab_reports"]
    if imaging_json is None:
        imaging_json = paths_config["data_files"]["imaging_reports"]
    if pathology_json is None:
        pathology_json = paths_config["data_files"]["pathology_reports"]
    if mutation_json is None:
        mutation_json = paths_config["data_files"]["mutation_reports"]
    if trials_json_path is None:
        trials_json_path = paths_config["data_files"]["trials"]
    
    # --- Visualization switches (no functional impact) ---
    visual = VisualConfig(
        enable=True,
        show_tables=True,
        show_rag_table=True,
        show_token_budget=False,
    )

    # Trace collection is always ON
    trace = TraceLogger(enabled=True)
    trace.emit("pipeline_start", {"visit_time": str(time) if time else None, "meta_info": str(meta_info)})

    print_section("MDT PIPELINE", "Observability ON (always)")

    client = args.client
    print(f"{Color.OKBLUE}{Color.BOLD}🕒 Query Time: {time}{Color.RESET}")

    # Normalize question (supports dict/list/str) and compute stable CASE fingerprint
    question_str = question_to_text(question)
    case_fingerprint = hashlib.sha1(question_str.encode("utf-8")).hexdigest()[:12]
    case_json = safe_load_case_json(question_str)

    print(f"{Color.OKBLUE}{Color.BOLD}🧾 CASE_FINGERPRINT: {case_fingerprint}{Color.RESET}")
    trace.emit("case_fingerprint", {"case_fingerprint": case_fingerprint})

    ###########################################################################
    # LOAD REPORTS (unchanged logic except already improved date handling)
    ###########################################################################
    print_section("1) Load Clinical Reports")

    cutoff_dt = make_cutoff(time, days_after=1)
    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S") if cutoff_dt else "None"
    print(f"{Color.OKBLUE}{Color.BOLD}⏱️  CUTOFF_DT (time + 1d): {cutoff_str}{Color.RESET}")
    print(meta_info,labs_json)
    lab_timeline_raw, lab_reports = load_patient_labs(meta_info, labs_json)
    
    im_timeline_raw, im_reports = load_patient_imaging(meta_info, imaging_json)

    path_timeline_raw, path_reports = [], []
    if pathology_json:
        path_timeline_raw, path_reports = load_patient_pathology(meta_info, pathology_json)

    mut_reports: List[Dict[str, Any]] = []
    if meta_info and mutation_json:
        mut_reports = load_patient_mutations(meta_info, mutation_json)

    try:
        print(f"{Color.OKCYAN}{Color.BOLD}[LAB] before filter: {report_range(lab_reports, 'report_date')}{Color.RESET}")
        print(f"{Color.OKCYAN}{Color.BOLD}[IMG] before filter: {report_range(im_reports, 'report_date')}{Color.RESET}")
        if pathology_json:
            print(f"{Color.OKCYAN}{Color.BOLD}[PATH] before filter: {report_range(path_reports, 'report_date')}{Color.RESET}")
        if mut_reports:
            print(f"{Color.OKCYAN}{Color.BOLD}[MUT] before filter: {report_range(mut_reports, 'report_date')}{Color.RESET}")
    except:
        pass

    trace.emit("reports_loaded", {
        "lab_n": len(lab_reports),
        "img_n": len(im_reports),
        "path_n": len(path_reports) if pathology_json else 0,
        "mut_n": len(mut_reports) if mut_reports else 0,
        "cutoff_dt": cutoff_str,
    })

    if cutoff_dt is not None:
        lab_reports = filter_before(lab_reports, "report_date", cutoff_dt)
        im_reports = filter_before(im_reports, "report_date", cutoff_dt)
        path_reports = filter_before(path_reports, "report_date", cutoff_dt)
        mut_reports = filter_before(mut_reports, "report_date", cutoff_dt)

    # rebuild fresh timelines
    lab_timeline = build_lab_timeline(lab_reports)
    im_timeline = build_imaging_timeline(im_reports)
    path_timeline = build_pathology_timeline(path_reports) if pathology_json else []

    ###########################################################################
    # REPORT SELECTION
    ###########################################################################
    context: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "lab": {},
        "imaging": {},
        "pathology": {},
        "mutation": {},
    }

    print_section("2) Report Selection (per Role)")
    for role in ROLES:
        print(f"{Color.OKBLUE}\n📄 Selecting reports for {role}...{Color.RESET}")

        selector = Agent(
            instruction=(
                "You are a report-filtering module in an MDT pipeline.\n"
                "Goal: select the MINIMAL set of reports NECESSARY for current decision-making.\n"
                "Prefer newest; max 3. JSON only."
            ),
            role=f"{role}_selector",
            model_info=model,
            client=client,
            max_tokens=220,
            max_prompt_tokens=2000,
        )

        if ROLE_PERMISSIONS[role]["lab"]:
            context["lab"][role] = expert_select_reports(selector, role, lab_timeline, lab_reports, "lab")
        else:
            context["lab"][role] = []

        if ROLE_PERMISSIONS[role]["imaging"]:
            context["imaging"][role] = expert_select_reports(selector, role, im_timeline, im_reports, "imaging")
        else:
            context["imaging"][role] = []

        if ROLE_PERMISSIONS[role]["pathology"] and pathology_json:
            context["pathology"][role] = expert_select_reports(selector, role, path_timeline, path_reports, "pathology")
        else:
            context["pathology"][role] = []

        if role in ("chair", "oncologist", "pathologist") and mut_reports:
            context["mutation"][role] = [
                {
                    "report_id": str(r.get("report_id")),
                    "date": (r.get("report_date") or "")[:19],
                    "raw_text": r.get("raw_text") or "",
                }
                for r in mut_reports
            ]
        else:
            context["mutation"][role] = []

    print(f"{Color.OKCYAN}{Color.BOLD}\n🧩 Selected reports:{Color.RESET}")
    print(json.dumps(summarize_selected_reports(context), ensure_ascii=False, indent=2))

    trace.emit("reports_selected", summarize_selected_reports(context))
    if visual.show_tables:
        print_selected_reports_table(context, roles=ROLES)

    ###########################################################################
    # GLOBAL GUIDELINE RAG
    ###########################################################################
    print_section("3) Global Guideline RAG")
    rag_query_builder = Agent(
        instruction="Construct concise English MDT guideline query.",
        role="rag_query_builder",
        model_info=model,
        client=client,
        max_tokens=120,
        max_prompt_tokens=2000,
    )
    
    rag_query = build_rag_query_for_mdt(rag_query_builder, question_str)
    print("rag_query",rag_query)
    
    # Use RAG config from paths config
    rag_config = paths_config["rag_store"]
    index_dir = rag_config["index_dir_template"].format(role="chair")
    embedding_model = rag_config.get("embedding_model", "BAAI/bge-m3")
    
    rag_pack, rag_raw = rag_search_pack(
        query=rag_query,
        index_dir=index_dir,
        model_name=embedding_model,
        device=device,
        topk=topk,
        collection_name="chair_chunks",
    )
    
    trace.emit("rag_query", {"query": rag_query})
    trace.emit("rag_hits", {"topk": topk, "n": len(rag_raw or [])})
    if visual.enable and visual.show_tables and visual.show_rag_table:
        print_rag_hits_table(rag_raw)

    guideline_digester = Agent(
        instruction="Digest guideline chunks into <=8 evidence bullets; no patient facts.",
        role="global_guideline_digester",
        model_info=model,
        client=client,
        max_tokens=4000,
        max_prompt_tokens=3500,
    )
    global_guideline_digest = summarize_rag_evidence(guideline_digester, rag_pack)
    print("rag_query",global_guideline_digest)
    ###########################################################################
    # INIT SPECIALIST AGENTS
    ###########################################################################
    agents = {
        role: init_expert_agent(
            role=role,
            question=question_str,
            model=model,
            client=client,
            context=context,
            case_fingerprint=case_fingerprint,
            global_guideline_digest=global_guideline_digest,
            device=device,
            topk=topk,
            visit_time=str(time) if time else None,
        )
        for role in ROLES
    }

    assistant = Agent(
        instruction="You are MDT assistant. Summarize only. Do not decide treatment.",
        role="assistant",
        model_info=model,
        client=client,
        max_tokens=5000,
        max_prompt_tokens=4500,
    )

    ###########################################################################
    # MDT DISCUSSION
    ###########################################################################
    print_section("4) MDT Discussion Engine")
    trace.emit("mdt_discussion_start", {"num_rounds": 2, "num_turns": 2})

    initial_ops, merged, final_round_ops, interaction_log = run_mdt_discussion(
        agents=agents,
        assistant=assistant,
        num_rounds=2,
        num_turns=2,
        visit_time=str(time) if time else None,
    )
    trace.emit("mdt_discussion_end", {"merged_chars": len(merged or "")})

    print_interaction_matrix(interaction_log, roles_order=ROLES)

    ###########################################################################
    # Assistant Clinical Trial Matching
    ###########################################################################
    trial_note = ""
    trial_agent = None
    if trials_json_path and os.path.exists(trials_json_path):
        try:
            with open(trials_json_path, "r", encoding="utf-8") as f:
                trials_list = json.load(f)

            print(f"{Color.OKBLUE}{Color.BOLD}\n[Assistant] Checking clinical trials...{Color.RESET}")
            trace.emit("trial_matching_start", {"trials_json_path": trials_json_path})

            # Use a dedicated agent for trial matching to avoid interference from the summarizer assistant.
            trial_agent = Agent(
                instruction=(
                    "You are an MDT assistant for clinical trial matching in gynecologic oncology. "
                    "Follow the trial recommendation gate strictly and recommend at most ONE trial."
                ),
                role="trial_selector",
                model_info=model,
                client=client,
                max_tokens=10000,
                max_prompt_tokens=4500,
                enable_local_log=True,
            )

            # IMPORTANT: Pass the normalized case string (question_str) for trial matching
            trial_note = assistant_trial_suggestion(
                agent=trial_agent,
                case_json_str=question_str,
                trials_list=trials_list,
            )

            # Debug: confirm the model call happened
            print(f"{Color.OKGREEN}✔ Trial selector local_log turns: {len(trial_agent.local_log)}{Color.RESET}")
            print(trial_note)
            trace.emit("trial_matching_end", {"recommended": "None" not in (trial_note or "")})

        except Exception as e:
            print(f"{Color.FAIL}Failed to process clinical trials: {e}{Color.RESET}")
            trial_note = ""

    ###########################################################################
    # FINAL OUTPUT
    ###########################################################################
    # Inject clinical trial suggestion if available
    if trial_note:
        agents["chair"].inject_assistant(
            f"[Assistant Clinical Trial Suggestion]\n{trial_note.strip()}"
        )
    print_section("5) Final Chair Output")
    trace.emit("final_output_start", {})
    print(f"{Color.BOLD}{Color.OKBLUE}\n📘 Generating final MDT output...{Color.RESET}")
    final_output = generate_final_output(
        chair_agent=agents["chair"],
        all_round_ops=final_round_ops,
        clinic_time=time
    )
    print(final_output)
    trace.emit("final_output_end", {"final_output_chars": len(final_output or "")})


    # Optionally append trial note to log or final output storage if needed

    agent_logs = {role: ag.local_log for role, ag in agents.items()}
    agent_logs["assistant"] = assistant.local_log
    if trial_agent is not None:
        agent_logs["trial_selector"] = trial_agent.local_log
    log_paths = save_mdt_log(
        question=question_str,
        final_output=final_output,
        initial_ops=initial_ops,
        merged=merged,
        final_round_ops=final_round_ops,
        interaction_log=interaction_log,
        agent_logs=agent_logs,
        trace_events=trace.events,
        trace_mermaid=trace.to_mermaid_flow() if trace.enabled else "",
    )

    try:
        # Use log directory from config
        log_dir = paths_config["output_dirs"]["mdt_logs"]
        save_case_html_report(
            log_dir=log_dir,
            ts=(log_paths or {}).get("ts") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            question_str=question_str,
            final_output=final_output,
            context=context,
            rag_query=rag_query,
            rag_pack=rag_pack,
            rag_raw=rag_raw,
            global_guideline_digest=global_guideline_digest,
            interaction_log=interaction_log,
            question_raw=question_raw,
            initial_ops=initial_ops,
            final_round_ops=final_round_ops,
            trace_events=trace.events,
            trace_mermaid=trace.to_mermaid_flow() if trace.enabled else "",
            roles_order=ROLES,
        )
    except Exception as e:
        print(f"{Color.WARNING}⚠ HTML report generation failed: {e}{Color.RESET}")

    print(f"{Color.BOLD}{Color.OKGREEN}\n=== MDT Multi-Expert Pipeline End ==={Color.RESET}")
    return final_output
