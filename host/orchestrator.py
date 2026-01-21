# agent_omgs.py
# NOTE: This file is part of the OMGs/MDT agent pipeline.
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
from utils.console_utils import Color, normalize_trial_compact, safe_parse_json_block, question_to_text, preview_text, print_prompt_budget
from servers.trace import VisualConfig, TraceLogger, print_selected_reports_table, print_section, print_rag_hits_table, warn_missing_evidence_tags
from core.agent import AgentError
from utils.error_handling import safe_agent_call, get_fallback_response
from servers.reporters import save_mdt_log, save_case_html_report
from utils.time_utils import make_cutoff, parse_dt, safe_date10, filter_before, report_range
from utils.time_utils import build_lab_timeline, build_imaging_timeline, build_pathology_timeline
from utils.stats_collector import collect_pipeline_stats
from servers.reports_selector import (
    load_patient_labs, load_patient_imaging, load_patient_pathology, 
    load_patient_mutations, summarize_selected_reports, select_reports_for_roles,
    expert_select_reports
)
from servers.evidence_search import build_rag_query_for_mdt, summarize_rag_evidence
from host.experts import ROLES, ROLE_PERMISSIONS, init_expert_agent
from servers.info_delivery import safe_load_case_json
from host.decision import generate_final_output, assistant_trial_suggestion, build_enhanced_case_for_trial, append_references_to_output
from core import Agent, init_client, get_paths_config, get_mdt_prompts
# Public API of `utils`
import random
import hashlib
import tiktoken
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from prettytable import PrettyTable, ALL
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################################
# 6. MULTI-ROUND MDT DISCUSSION ENGINE
###############################################################################
def run_mdt_discussion(
    agents: Dict[str, "Agent"],
    assistant: "Agent",
    num_rounds: int = 2,
    num_turns: int = 2,
    max_merged_chars: int = 10000,
    max_turn_delta_chars: int = 900,
    max_targets_per_speaker: int = 4,
    visit_time: Optional[str] = None,
    trace: Optional["TraceLogger"] = None,
) -> Tuple[Dict[str, str], str, Dict[str, Dict[str, str]], Dict[str, Dict[str, Dict[str, Dict[str, Optional[str]]]]]]:
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

    def _clip_head(s: str, max_chars: int) -> str:
        if not s:
            return ""
        return s if len(s) <= max_chars else s[:max_chars]
    
    # pack the context of the MDT：
    # (memory……)
    # [RECENT_DELTAS]
    # (deltas……)
    def _pack_context(memory: str, deltas: str, max_chars: int, memory_ratio: float = 0.75) -> str:
        """Keep structured memory in the FRONT, and keep only recent deltas in the TAIL."""
        memory = memory or ""
        deltas = deltas or ""
        if not deltas:
            return _clip_head(memory, max_chars)

        sep = "\n\n[RECENT_DELTAS]\n"
        # Allocate budget: preserve memory first
        mem_budget = max(200, int(max_chars * memory_ratio))
        # Remaining budget goes to deltas (+separator)
        delta_budget = max(0, max_chars - min(len(memory), mem_budget) - len(sep))

        mem_part = _clip_head(memory, mem_budget)
        delta_part = _clip_tail(deltas, delta_budget)
        if not delta_part:
            return _clip_head(mem_part, max_chars)
        return _clip_head(mem_part + sep + delta_part, max_chars)

    # Load MDT prompts from config
    mdt_prompts = get_mdt_prompts().get("mdt_discussion", {})
    
    print(f"{Color.OKCYAN}{Color.BOLD}🧠 Starting MDT Discussion Engine...{Color.RESET}")
    agent_list = list(agents.keys())

    emoji_pool = ["👨‍⚕️","👩‍⚕️","🧑‍⚕️","👨🏻‍⚕️","👩🏼‍⚕️","👨🏽‍⚕️","👩🏽‍⚕️","👨🏾‍⚕️","👩🏾‍⚕️","🧑🏾‍⚕️"]
    random.shuffle(emoji_pool)
    role_to_emoji = {r: emoji_pool[i % len(emoji_pool)] for i, r in enumerate(agent_list)}
    chair_role = "chair" if "chair" in agent_list else agent_list[0]

    last_msg_by_pair = {} # record the last message by pair to avoid duplicate messages from the same speaker-target pair, which helps reduce redundancy but may suppress necessary repeated clarifications in some scenarios; adjust based on actual usage.

    # INITIAL OPINIONS
    print(f"{Color.BOLD}{Color.OKBLUE}\n📌 Collecting Initial Opinions...{Color.RESET}")
    initial_ops = {} # we will use this to store the initial opinions of the experts
    initial_opinion_prompt = mdt_prompts.get("initial_opinion", 
        "Give INITIAL opinion (use ONLY your system-provided patient facts).\n"
        "Return up to 3 bullets, each ≤20 words.\n"
        "If key data missing, say exactly what needs updating.\n"
        "At least ONE bullet must be evidence-based and include [@guideline:doc_id | Page xx] or [@pubmed | PMID].\n"
        "If you reference treatment strategy categories, guidelines, trials, or literature evidence, include tags [@guideline:doc_id | Page xx] or [@pubmed | PMID]."
    )
    # initial_ops is a dictionary that stores the initial opinions of the experts
    for i, (role, ag) in enumerate(agents.items(), start=1):
        print(f"{Color.OKGREEN} [{i}/{len(agents)}] {role}:{Color.RESET}")
        if trace:
            trace.emit("mdt_initial_opinion_role_start", {"role": role, "order": i})
        print_prompt_budget(f"{role}/initial", initial_opinion_prompt) # print the token budget of the initial opinion prompt
        
        # Use safe_agent_call for error handling
        try:
            op = ag.chat(initial_opinion_prompt) # get the initial opinion of the expert
            if trace:
                trace.emit("mdt_initial_opinion_role_end", {"role": role, "chars": len(op or '')})
            print(f"{Color.OKCYAN}   {role_to_emoji[role]} {op}{Color.RESET}")
            warn_missing_evidence_tags(op, role=f"{role}/initial", trace=trace) # warn the missing evidence tags in the initial opinion
            initial_ops[role] = op # store the initial opinion of the expert in the initial_ops dictionary
        except AgentError as e:
            error_msg = get_fallback_response(role, "initial_opinion")
            print(f"{Color.WARNING}[WARNING] {role} failed: {e.original_error}{Color.RESET}")
            print(f"{Color.OKCYAN}   {role_to_emoji[role]} {error_msg}{Color.RESET}")
            initial_ops[role] = error_msg
            if trace:
                trace.emit("agent_error", {
                    "role": role,
                    "stage": "initial_opinion",
                    "error": str(e.original_error),
                    "error_type": type(e.original_error).__name__
                })
        except Exception as e:
            # Catch any unexpected exceptions
            error_msg = get_fallback_response(role, "initial_opinion")
            print(f"{Color.FAIL}[ERROR] {role} unexpected error: {e}{Color.RESET}")
            print(f"{Color.OKCYAN}   {role_to_emoji[role]} {error_msg}{Color.RESET}")
            initial_ops[role] = error_msg
            if trace:
                trace.emit("agent_error", {
                    "role": role,
                    "stage": "initial_opinion",
                    "error": str(e),
                    "error_type": "UnexpectedException"
                })

    summarize_template = mdt_prompts.get("summarize_initial_template",
        "Summarize expert opinions concisely for MDT.\n{opinions}\n\n"
        "Output:\nKey Knowledge:\n- ...\nControversies:\n- ...\nMissing Info:\n- ...\nWorking Plan:\n- ..."
    )
    # summarize the initial opinions of the experts
    # merged is the structured memory of the MDT
    fallback_merged = f"Key Knowledge:\n{json.dumps(initial_ops, ensure_ascii=False, indent=2)}\n\nControversies:\n- To be determined\n\nMissing Info:\n- To be determined\n\nWorking Plan:\n- To be determined"
    
    merged = safe_agent_call(
        agent=assistant,
        prompt=summarize_template.format(opinions=json.dumps(initial_ops, ensure_ascii=False, separators=(',', ':'))),
        role="assistant",
        stage="initial_summary",
        fallback=fallback_merged,
        trace=trace,
        max_retries=3  # Retry up to 3 times for rate limits (429 errors) with exponential backoff
    )
    # Structured MDT memory (always kept at the front)
    memory_state = _clip_head(merged, max_merged_chars)
    # Rolling discussion deltas (kept as a tail window)
    delta_state = ""
    # delta_state is the key of the turn continuity.
    # delta_State is continuous, it is the delta of the MDT.
    merged = _pack_context(memory_state, delta_state, max_merged_chars)
    print("merged:\n", merged)

    interaction_log = {
        f"Round {r}": {
            f"Turn {t}": {s: {d: None for d in agent_list} for s in agent_list}
            for t in range(1, num_turns + 1)
        }
        for r in range(1, num_rounds + 1)
    }
    final_round_ops = {}

    # Load prompt templates
    round_summary_template = mdt_prompts.get("round_summary_template",
        "MDT global knowledge:\n{merged}\n\nRe-summarize concisely. Must include:\n"
        "Key Knowledge:\n- ...\nControversies:\n- ...\nMissing Info:\n- ...\nWorking Plan:\n- ..."
    )
    speak_prompt_template = mdt_prompts.get("speak_prompt_template",
        "ROLE: {role}. VISIT: {visit_time}\n"
        "Default is NOT to speak. Speak ONLY if: conflict | safety | missing-critical | new-critical.\n\n"
        "CONTEXT (latest):\n{context}\n\n"
        "Allowed targets: [{allowed_targets}]\n"
        'Return ONE-LINE JSON only:{{"speak":"yes/no","messages":[{{"target":"<role>","message":"<1-2 sentences>","why":"conflict|safety|missing|new"}}]}}'
    )

    # MAIN DISCUSSION
    # Rounds
    for r in range(1, num_rounds + 1):
        print(f"{Color.WARNING}{Color.BOLD}\n==================== ROUND {r} ===================={Color.RESET}")
        round_key = f"Round {r}"

        # Re-summarize ONLY the structured memory. Deltas stay separate.
        # if r == 1, we use the memory_state as the summary
        # otherwise, we use the assistant to summarize the memory_state
        if r == 1:
            summary = memory_state
        else:
            summary = safe_agent_call(
                agent=assistant,
                prompt=round_summary_template.format(merged=memory_state),
                role="assistant",
                stage=f"round_{r}_summary",
                fallback=memory_state,  # Fallback: keep existing memory_state
                trace=trace,
                max_retries=3  # Retry up to 3 times for rate limits (429 errors)
            )
        memory_state = _clip_head(f"[MDT_GLOBAL_KNOWLEDGE]\n{summary}", max_merged_chars)
        merged = _pack_context(memory_state, delta_state, max_merged_chars)

        MDT_should_stop = False
        # Turns
        for t in range(1, num_turns + 1):
            print(f"{Color.BOLD}{Color.OKCYAN}\n--- Turn {t} ---{Color.RESET}")
            turn_key = f"Turn {t}"
            num_speakers = 0
            turn_msgs_compact = []
            merged = _pack_context(memory_state, delta_state, max_merged_chars)
            ctx_for_turn = merged

            for role, ag in agents.items():
                allowed_targets = [x for x in agent_list if x != role]
                allowed_targets_str = ",".join(allowed_targets)

                speak_prompt = speak_prompt_template.format(
                    role=role,
                    visit_time=visit_time or 'Unknown',
                    context=ctx_for_turn,
                    allowed_targets=allowed_targets_str
                )

                try:
                    resp = ag.chat(speak_prompt)
                    data = safe_parse_json_block(resp)
                except AgentError as e:
                    # Skip this agent's turn if it fails
                    if trace:
                        trace.emit("agent_error", {
                            "role": role,
                            "stage": f"turn_{t}",
                            "error": str(e.original_error),
                            "error_type": type(e.original_error).__name__
                        })
                    continue
                except Exception as e:
                    # Skip this agent's turn for unexpected errors
                    print(f"{Color.WARNING}[WARNING] {role} failed to speak in turn {t}: {e}{Color.RESET}")
                    if trace:
                        trace.emit("agent_error", {
                            "role": role,
                            "stage": f"turn_{t}",
                            "error": str(e),
                            "error_type": "UnexpectedException"
                        })
                    continue

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
                    
                    # Optional: Warn if message mentions evidence but lacks tags
                    # (Not enforced, just a helpful reminder)
                    if trace:
                        warn_missing_evidence_tags(msg, role=f"{role}->{target}/turn_{t}", trace=trace)

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
                # Append into rolling delta buffer only
                delta_state = _append_bounded(delta_state, f"[R{r}T{t} DELTA] {delta_text}", max_merged_chars)
                merged = _pack_context(memory_state, delta_state, max_merged_chars)

        # FINAL plans for this round
        final_round_ops[round_key] = {}
        
        # Build compact discussion history for this round
        round_discussion_summary = []
        for t in range(1, num_turns + 1):
            turn_key = f"Turn {t}"
            turn_msgs = []
            for src_role in agent_list:
                for dst_role in agent_list:
                    if src_role != dst_role:
                        msg = interaction_log[round_key][turn_key].get(src_role, {}).get(dst_role)
                        if msg:
                            turn_msgs.append(f"{src_role}→{dst_role}: {msg}")
            if turn_msgs:
                round_discussion_summary.append(f"Turn {t}: {' | '.join(turn_msgs[:3])}")  # Limit to 3 messages per turn
        
        discussion_context = "\n".join(round_discussion_summary[-4:]) if round_discussion_summary else "No direct discussions in this round."
        
        final_plan_template = mdt_prompts.get("final_plan_template",
            "Given MDT context:\n{merged}\n\n"
            "DISCUSSION HISTORY (this round):\n{discussion_history}\n\n"
            "Provide FINAL refined plan based on the above context and discussions.\n"
            "Up to 3 bullets, each ≤20 words.\n"
            "Any factual claim must include [@actual_report_id | LAB/Genomics/MR/CT] using actual report_id from report data or say unknown.\n"
            "At least ONE bullet must be evidence-based and include [@guideline:doc_id | Page xx] or [@pubmed | PMID].\n"
            "If you reference treatment strategy categories, guidelines, trials, or literature evidence, include tags [@guideline:doc_id | Page xx] or [@pubmed | PMID].\n"
            "If discussions mentioned specific evidence, you may reference it with appropriate tags."
        )
        print(f"{Color.BOLD}{Color.OKBLUE}\n📘 Finalizing Expert Plans for ROUND {r} ...{Color.RESET}")
        for role, ag in agents.items():
            print(f"{Color.OKBLUE}{Color.BOLD} - {role}:{Color.RESET}")
            try:
                final_op = ag.chat(final_plan_template.format(merged=merged, discussion_history=discussion_context))
                print(f"{Color.OKGREEN}{final_op}{Color.RESET}\n")
                warn_missing_evidence_tags(final_op, role=f"{role}/final_round_{r}", trace=trace)
                final_round_ops[round_key][role] = final_op
            except AgentError as e:
                error_msg = get_fallback_response(role, "final_plan")
                print(f"{Color.WARNING}[WARNING] {role} final plan failed: {e.original_error}{Color.RESET}")
                print(f"{Color.OKGREEN}{error_msg}{Color.RESET}\n")
                final_round_ops[round_key][role] = error_msg
                if trace:
                    trace.emit("agent_error", {
                        "role": role,
                        "stage": f"final_plan_round_{r}",
                        "error": str(e.original_error),
                        "error_type": type(e.original_error).__name__
                    })
            except Exception as e:
                error_msg = get_fallback_response(role, "final_plan")
                print(f"{Color.FAIL}[ERROR] {role} final plan unexpected error: {e}{Color.RESET}")
                print(f"{Color.OKGREEN}{error_msg}{Color.RESET}\n")
                final_round_ops[round_key][role] = error_msg
                if trace:
                    trace.emit("agent_error", {
                        "role": role,
                        "stage": f"final_plan_round_{r}",
                        "error": str(e),
                        "error_type": "UnexpectedException"
                    })

        # IMPORTANT: Round FINAL plans must update the structured memory, not the delta window.
        round_final_pack = json.dumps(final_round_ops[round_key], ensure_ascii=False, separators=(',', ':'))
        fallback_memory = _clip_head(
            memory_state + "\n\n" + f"[ROUND {r} FINAL_PLANS] {round_final_pack}",
            max_merged_chars,
        )
        
        memory_update = safe_agent_call(
            agent=assistant,
            prompt=(
                "You are MDT assistant. Update MDT GLOBAL structured memory by integrating ROUND FINAL plans. "
                "Keep the same output format with: Key Knowledge / Controversies / Missing Info / Working Plan.\n\n"
                f"CURRENT_MDT_GLOBAL_KNOWLEDGE:\n{memory_state}\n\n"
                f"ROUND_{r}_FINAL_PLANS_JSON:\n{round_final_pack}"
            ),
            role="assistant",
            stage=f"memory_update_round_{r}",
            fallback=fallback_memory,  # Fallback: preserve final plans in memory
            trace=trace,
            max_retries=3  # Retry up to 3 times for rate limits (429 errors)
        )
        memory_state = _clip_head(memory_update, max_merged_chars)

        # Start next round with a clean delta window
        delta_state = ""
        merged = _pack_context(memory_state, delta_state, max_merged_chars)

        if MDT_should_stop:
            print(f"{Color.WARNING}{Color.BOLD}🚫 MDT stopped early after Round {r}. No further rounds will be executed.{Color.RESET}")
            return initial_ops, merged, final_round_ops, interaction_log
        print("initial_ops:\n",initial_ops)
        print("merged:\n",merged)
        print("final_round_ops:\n",final_round_ops)
        # print(interaction_log)
    return initial_ops, merged, final_round_ops, interaction_log

# important for evidence search!!!!
def _build_rag_key_facts(case_json: Dict[str, Any], mut_reports: List[Dict[str, Any]]) -> str:
    """Build KEY FACTS string for RAG query from case data.
    
    Simply includes raw mutation report text directly - let LLM parse it.
    """
    parts: List[str] = []
    case_core = case_json.get("CASE_CORE") or {}
    diagnosis = case_core.get("DIAGNOSIS") or {}
    if diagnosis:
        primary = diagnosis.get("primary") or "Unknown"
        hist = diagnosis.get("histology") or "Unknown"
        comps = diagnosis.get("components") or []
        comp_txt = f" components={';'.join([str(x) for x in comps])}" if comps else ""
        parts.append(f"DIAGNOSIS: primary={primary}; histology={hist};{comp_txt}")
    pathology = case_json.get("PATHOLOGY") or {}
    specimens = pathology.get("specimens") if isinstance(pathology, dict) else None
    if isinstance(specimens, list) and specimens:
        diag = specimens[0].get("diagnosis") or ""
        if diag:
            parts.append(f"PATHOLOGY: {preview_text(diag, 160)}")
    plat = case_core.get("PLATINUM_STATUS_CURRENT") or case_core.get("PLATINUM_STATUS")
    pfi = case_core.get("PLATINUM_PFI_CURRENT") or case_core.get("PFI_days")
    if plat or pfi:
        parts.append(f"PLATINUM: status={plat or 'Unknown'}; pfi_days={pfi or 'Unknown'}")
    
    # Only use GENETICS from case_core if NO mutation reports are available
    # If mutation reports exist, they are the source of truth - don't use case_core values
    # This prevents using "not reported" or "Unknown" from case_core when actual reports exist
    # important for evidence search!!!! must be before mutation reports are included
    if not mut_reports:
        hrd = case_core.get("HRD")
        brca1 = case_core.get("BRCA1")
        brca2 = case_core.get("BRCA2")
        if any([hrd, brca1, brca2]):
            parts.append(f"GENETICS: HRD={hrd or 'Unknown'}; BRCA1={brca1 or 'Unknown'}; BRCA2={brca2 or 'Unknown'}")
    
    biomarkers = case_core.get("BIOMARKERS") or {}
    if biomarkers:
        keys = ["CA125", "HE4", "CA19-9", "CA15-3", "AFP", "CEA", "TMB", "MSI", "PDL1_CPS"]
        items = [f"{k}={biomarkers.get(k)}" for k in keys if biomarkers.get(k)]
        if items:
            parts.append("BIOMARKERS: " + "; ".join(items[:6]))
    
    # Include full mutation report raw_text directly - let LLM parse it
    # This takes precedence over case_core GENETICS values
    # !!!!important; mut_reports is gold criteria for evidence search!!!!
    if mut_reports:
        latest = mut_reports[-1]
        rid = latest.get("report_id") or "Unknown"
        rdate = latest.get("report_date") or ""
        raw = latest.get("raw_text") or ""
        # Include full text (up to 3000 chars to avoid token bloat, but should cover most reports)
        raw_text = preview_text(raw, 3000) if raw else ""
        parts.append(f"MUTATION_REPORT: id={rid}; date={str(rdate)[:10]}; full_text={raw_text}")
    
    return "\n".join(parts)


###############################################################################
# INTERACTION DIRECTION MATRIX（PrettyTable）
###############################################################################
def _count_interactions(
    interaction_log: Dict[str, Dict[str, Dict[str, Dict[str, Optional[str]]]]],
    src: str,
    dst: str
) -> int:
    """Count total interactions from src to dst across all rounds and turns."""
    c = 0
    for rnd in interaction_log.values():
        for turn in rnd.values():
            msg = turn.get(src, {}).get(dst)
            if msg:
                c += 1
    return c


def print_interaction_matrix(
    interaction_log: Dict[str, Dict[str, Dict[str, Dict[str, Optional[str]]]]],
    roles_order: List[str] = ROLES
) -> None:
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
###############################################################################
#  MAIN ENTRY
###############################################################################
###############################################################################
def process_omgs_multi_expert_query(
    question: Any,
    question_raw: Optional[str],
    model: str,
    args: Any,
    time: Optional[str] = None,
    meta_info: Optional[str] = None,
    labs_json: Optional[str] = None,
    imaging_json: Optional[str] = None,
    pathology_json: Optional[str] = None,
    mutation_json: Optional[str] = None,
    device: str = "auto",
    topk: int = 5,
    case_filter_buffer_days: int = 120,
    strict_context_prune: bool = False,
    trials_json_path: Optional[str] = None
) -> str:
    print(f"{Color.BOLD}{Color.OKGREEN}\n=== MDT Multi-Expert Pipeline Start ==={Color.RESET}")
    
    # Record pipeline start time for statistics
    pipeline_start_time = datetime.now()
    
    # Load paths configuration
    paths_config = get_paths_config()
    # print(paths_config)
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
    
    # Load reports with error handling - each load is independent
    # If a file is missing, the function returns empty lists, so we can continue
    try:
        lab_timeline_raw, lab_reports = load_patient_labs(meta_info, labs_json)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to load lab reports: {e}. Continuing with empty lab data.{Color.RESET}")
        lab_timeline_raw, lab_reports = [], []
    
    try:
        im_timeline_raw, im_reports = load_patient_imaging(meta_info, imaging_json)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to load imaging reports: {e}. Continuing with empty imaging data.{Color.RESET}")
        im_timeline_raw, im_reports = [], []

    path_timeline_raw, path_reports = [], []
    if pathology_json:
        try:
            path_timeline_raw, path_reports = load_patient_pathology(meta_info, pathology_json)
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to load pathology reports: {e}. Continuing with empty pathology data.{Color.RESET}")
            path_timeline_raw, path_reports = [], []

    mut_reports: List[Dict[str, Any]] = []
    if meta_info and mutation_json:
        try:
            mut_reports = load_patient_mutations(meta_info, mutation_json)
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to load mutation reports: {e}. Continuing with empty mutation data.{Color.RESET}")
            mut_reports = []

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
    context = select_reports_for_roles(
        roles=ROLES,
        role_permissions=ROLE_PERMISSIONS,
        lab_timeline=lab_timeline,
        lab_reports=lab_reports,
        im_timeline=im_timeline,
        im_reports=im_reports,
        path_timeline=path_timeline,
        path_reports=path_reports,
        mut_reports=mut_reports,
        pathology_json=pathology_json,
        agent_class=Agent,
        expert_select_fn=expert_select_reports,
        model=model,
        client=client,
        color=Color,
        use_shared=True,
    )

    print(f"{Color.OKCYAN}{Color.BOLD}\n🧩 Selected reports:{Color.RESET}")
    print(json.dumps(summarize_selected_reports(context), ensure_ascii=False, indent=2))

    trace.emit("reports_selected", summarize_selected_reports(context))
    if visual.show_tables:
        print_selected_reports_table(context, roles=ROLES)

    ###########################################################################
    # GLOBAL GUIDELINE RAG
    ###########################################################################
    print_section("3) Guideline + PubMed RAG")
    # Load agent prompts from config
    agent_prompts = get_mdt_prompts().get("agents", {})
    
    rag_query_builder = Agent(
        instruction=agent_prompts.get("rag_query_builder", 
            "Construct concise English MDT guideline query."),
        role="rag_query_builder",
        model_info=model,
        client=client,
        max_tokens=5000,
        max_prompt_tokens=20000,
    )
    
    rag_key_facts = _build_rag_key_facts(case_json, mut_reports)
    trace.emit("rag_key_facts", {"facts": rag_key_facts})
    
    # Build RAG query with error handling
    # IMPORTANT: If mutation reports exist, inject HRD/BRCA values into case_json
    # to override "Unknown" values that confuse the LLM
    rag_question_str = question_str  # Default to original
    if mut_reports:
        import copy
        import re
        rag_case_json = copy.deepcopy(case_json)
        latest_mut = mut_reports[-1]
        raw_text = latest_mut.get("raw_text", "")
        if raw_text:
            # Extract HRD status
            if "HRD" in raw_text:
                if "阴性" in raw_text or "negative" in raw_text.lower():
                    rag_case_json.setdefault("CASE_CORE", {})["HRD"] = "Negative"
                elif "阳性" in raw_text or "positive" in raw_text.lower():
                    rag_case_json.setdefault("CASE_CORE", {})["HRD"] = "Positive"
            # Extract BRCA1 status
            if "BRCA1" in raw_text:
                if any(x in raw_text for x in ["未检出", "阴性", "视为阴性"]):
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = "Negative"
                elif "突变" in raw_text and "致病" in raw_text:
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA1"] = "Positive"
            # Extract BRCA2 status
            if "BRCA2" in raw_text:
                if any(x in raw_text for x in ["未检出", "阴性", "视为阴性"]):
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = "Negative"
                elif "突变" in raw_text and "致病" in raw_text:
                    rag_case_json.setdefault("CASE_CORE", {})["BRCA2"] = "Positive"
        rag_question_str = json.dumps(rag_case_json, ensure_ascii=False)
    
    try:
        rag_query = build_rag_query_for_mdt(rag_query_builder, rag_question_str, key_facts=rag_key_facts)
        print("rag_query",rag_query)
    except Exception as e:
        # Fallback: use simplified query from case JSON
        print(f"{Color.WARNING}[WARNING] RAG query builder failed: {e}{Color.RESET}")
        case_core = case_json.get("CASE_CORE", {}) or {}
        diagnosis = case_core.get("DIAGNOSIS", {}) or {}
        primary = diagnosis.get("primary", "ovarian cancer")
        rag_query = f"{primary} treatment guidelines"
        if trace:
            trace.emit("pipeline_error", {
                "stage": "rag_query_build",
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback_used": True
            })
    
    # Use global guideline RAG (respects config: use_per_role_rag / default_role)
    from servers.evidence_search import (
        get_global_guideline_rag,
        pubmed_search_pack,
        merge_rag_packs,
        merge_rag_raw,
    )
    
    # RAG retrieval with error handling
    # If RAG fails due to network issues, skip it and continue
    try:
        guideline_pack, guideline_raw = get_global_guideline_rag(
            question=rag_query,
            device=device,
            topk=topk,
        )
        # Check if initialization failed
        if guideline_pack == "(RAG: initialization failed)":
            print(f"{Color.WARNING}[WARNING] Guideline RAG initialization failed (likely network issue). Skipping RAG retrieval.{Color.RESET}")
            guideline_raw = []
            if trace:
                trace.emit("pipeline_error", {
                    "stage": "guideline_rag",
                    "error": "RAG initialization failed - network issue",
                    "error_type": "NetworkError",
                    "fallback_used": True
                })
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Guideline RAG retrieval failed: {e}. Skipping RAG retrieval.{Color.RESET}")
        guideline_pack = "(RAG: no evidence found)"
        guideline_raw = []
        if trace:
            trace.emit("pipeline_error", {
                "stage": "guideline_rag",
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback_used": True
            })
    
    try:
        pubmed_pack, pubmed_raw = pubmed_search_pack(
            query=rag_query,
            topk=5,
        )
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] PubMed RAG retrieval failed: {e}{Color.RESET}")
        pubmed_pack = "(PUBMED: no evidence found)"
        pubmed_raw = []
        if trace:
            trace.emit("pipeline_error", {
                "stage": "pubmed_rag",
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback_used": True
            })
    rag_pack = merge_rag_packs(guideline_pack, pubmed_pack)
    rag_raw = merge_rag_raw(guideline_raw, pubmed_raw)
    
    # Store RAG results in reference cache for later retrieval
    try:
        from utils.reference_cache import get_reference_cache
        ref_cache = get_reference_cache()
        ref_cache.store_rag_results(rag_raw)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] Failed to cache RAG references: {e}{Color.RESET}")
    
    trace.emit("rag_query", {"query": rag_query})
    trace.emit("rag_hits", {"source": "guideline", "topk": topk, "n": len(guideline_raw or [])})
    trace.emit("rag_hits", {"source": "pubmed", "topk": 5, "n": len(pubmed_raw or [])})
    trace.emit("rag_hits_merged", {"n": len(rag_raw or [])})
    if visual.enable and visual.show_tables and visual.show_rag_table:
        print_rag_hits_table(rag_raw)

    # Count RAG results for dynamic instruction (1:1 mapping: each RAG result gets one bullet)
    rag_count = len(rag_raw) if rag_raw else 0
    # Always use dynamic instruction to ensure 1:1 evidence mapping (ignore config static value)
    guideline_digester = Agent(
        instruction=f"Digest RAG chunks into exactly {rag_count} evidence bullets (one per RAG result); no patient facts.",
        role="global_guideline_digester",
        model_info=model,
        client=client,
        max_tokens=4000,
        max_prompt_tokens=3500,
    )
    
    # RAG evidence summarization with error handling
    try:
        global_guideline_digest = summarize_rag_evidence(guideline_digester, rag_pack, rag_raw=rag_raw)
        print("rag_digest", global_guideline_digest)
    except Exception as e:
        print(f"{Color.WARNING}[WARNING] RAG evidence summarization failed: {e}{Color.RESET}")
        # Fallback: use first 3 RAG results as digest
        if rag_raw and len(rag_raw) > 0:
            digest_lines = []
            for i, r in enumerate(rag_raw[:3], 1):
                source = r.get("source", "")
                if source == "guideline":
                    doc_id = r.get("doc_id", "")
                    page = r.get("page", "NA")
                    tag = f"[@guideline:{doc_id} | Page {page}]"
                elif source == "pubmed":
                    pmid = r.get("pmid", "")
                    tag = f"[@pubmed | {pmid}]"
                else:
                    tag = f"[unknown source {i}]"
                text = r.get("text", "") or r.get("abstract", "")
                preview = text[:200] + "..." if len(text) > 200 else text
                digest_lines.append(f"- {preview} {tag}")
            global_guideline_digest = "\n".join(digest_lines) if digest_lines else "# No RAG evidence available"
        else:
            global_guideline_digest = "# No RAG evidence available"
        if trace:
            trace.emit("pipeline_error", {
                "stage": "rag_summarization",
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback_used": True
            })
    
    ###########################################################################
    # INIT SPECIALIST AGENTS
    ###########################################################################
    agents = {}
    failed_roles = []
    for role in ROLES:
        try:
            agents[role] = init_expert_agent(
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
        except Exception as e:
            print(f"{Color.WARNING}[WARNING] Failed to initialize {role} agent: {e}{Color.RESET}")
            failed_roles.append(role)
            if trace:
                trace.emit("pipeline_error", {
                    "stage": "agent_init",
                    "role": role,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
    
    # If chair failed, use first successfully initialized agent as chair
    if "chair" in failed_roles and agents:
        first_role = list(agents.keys())[0]
        print(f"{Color.WARNING}[WARNING] Chair agent failed. Using {first_role} as fallback chair.{Color.RESET}")
        agents["chair"] = agents[first_role]
        failed_roles.remove("chair")
    
    # Ensure we have at least one agent
    if not agents:
        raise RuntimeError("Failed to initialize any expert agents. Cannot proceed with MDT discussion.")
    
    if failed_roles:
        print(f"{Color.WARNING}[WARNING] Some roles failed to initialize: {failed_roles}. Continuing with available agents.{Color.RESET}")

    assistant = Agent(
        instruction=agent_prompts.get("assistant", 
            "You are MDT assistant. Summarize only. Do not decide treatment."),
        role="assistant",
        model_info=model,
        client=client,
        max_tokens=10000,
        max_prompt_tokens=10000,
    )

    ###########################################################################
    # MDT DISCUSSION
    ###########################################################################
    print_section("4) MDT Discussion Engine")
    trace.emit("mdt_discussion_start", {"num_rounds": 2, "num_turns": 2})

    # run the MDT discussion engine
    initial_ops, merged, final_round_ops, interaction_log = run_mdt_discussion(
        agents=agents,
        assistant=assistant,
        num_rounds=1, # 2 for formal； 1 for test
        num_turns=1,  # 2
        visit_time=str(time) if time else None,
        trace=trace,
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
                instruction=agent_prompts.get("trial_selector",
                    "You are an MDT assistant for clinical trial matching in gynecologic oncology. "
                    "Follow the trial recommendation gate strictly and recommend at most ONE trial."
                ),
                role="trial_selector",
                model_info=model,
                client=client,
                max_tokens=20000,
                max_prompt_tokens=12000,  # Increased to accommodate both patient case and trials list
                enable_local_log=True,
            )

            # IMPORTANT: Build enhanced case information including pathology and mutation reports
            enhanced_case_info = build_enhanced_case_for_trial(
                case_json=case_json,
                path_reports=path_reports,
                mut_reports=mut_reports,
                question_str=question_str
            )
            # print("enhanced_case_info",enhanced_case_info)
            trial_note = assistant_trial_suggestion(
                agent=trial_agent,
                case_json_str=enhanced_case_info,
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
    print_section("5) Final Chair Output")
    trace.emit("final_output_start", {})
    print(f"{Color.BOLD}{Color.OKBLUE}\n📘 Generating final MDT output...{Color.RESET}")
    # print(merged)
    # print(initial_ops)
    # print(interaction_log)
    final_output = generate_final_output(
        chair_agent=agents["chair"],
        all_round_ops=final_round_ops,
        clinic_time=time,
        merged=merged,
        initial_ops=initial_ops,
        interaction_log=interaction_log,
        trial_note=trial_note,
        trace=trace
    )
    # Post-process: append References section with evidence details
    final_output = append_references_to_output(
        final_output,
        trial_note=trial_note,
        report_context=context,
    )
    print(final_output)
    warn_missing_evidence_tags(final_output, role="chair/final_output", trace=trace)
    trace.emit("final_output_end", {"final_output_chars": len(final_output or "")})

    # Record pipeline end time and collect statistics
    pipeline_end_time = datetime.now()
    db_path = paths_config["output_dirs"]["api_trace_db"]
    pipeline_stats = collect_pipeline_stats(pipeline_start_time, pipeline_end_time, db_path)
    # Add provider information if available
    if hasattr(args, 'client') and hasattr(args.client, 'provider'):
        pipeline_stats["provider"] = args.client.provider

    # Optionally append trial note to log or final output storage if needed
    agent_logs = {role: ag.local_log for role, ag in agents.items()}
    agent_logs["assistant"] = assistant.local_log
    if trial_agent is not None:
        agent_logs["trial_selector"] = trial_agent.local_log
    # Use log directory from config
    log_dir = paths_config["output_dirs"]["mdt_logs"]
    log_paths = save_mdt_log(
        question=question_str,
        final_output=final_output,
        initial_ops=initial_ops,
        merged=merged,
        final_round_ops=final_round_ops,
        interaction_log=interaction_log,
        agent_logs=agent_logs,
        log_dir=log_dir,
        trace_events=trace.events,
        trace_mermaid=trace.to_mermaid_flow() if trace.enabled else "",
    )

    try:
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
            trial_note=trial_note,
            initial_ops=initial_ops,
            final_round_ops=final_round_ops,
            trace_events=trace.events,
            trace_mermaid=trace.to_mermaid_flow() if trace.enabled else "",
            roles_order=ROLES,
            pipeline_stats=pipeline_stats,
        )
    except Exception as e:
        print(f"{Color.WARNING}⚠ HTML report generation failed: {e}{Color.RESET}")

    print(f"{Color.BOLD}{Color.OKGREEN}\n=== MDT Multi-Expert Pipeline End ==={Color.RESET}")
    return final_output
