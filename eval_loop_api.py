"""
eval_loop_api.py – Same functionality as eval_loop.py but uses the OpenAI API
for inference instead of vLLM local serving.

Supports: custom base URL, model name, reasoning effort, max tokens,
temperature, top_p, and concurrent request batching.
"""

from typing import List, Dict, Any, Optional, Callable
import argparse, json, math, os, re, random, pickle, asyncio
from pathlib import Path
from functools import partial

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from openai import AsyncOpenAI
import openai

from rewards.math import last_boxed_only_string, remove_boxed, is_equiv
from rewards.arxiv_math import parse_answer, extract_answer, check_answers
from reasoning_gym.factory import get_score_answer_fn


# --------------------- helpers ---------------------

def load_latest_loop_file(dir_path):
    dir_path = Path(dir_path)
    pattern = re.compile(r"loop_(\d+)\.pkl$")
    max_i = -1
    latest_file = None
    for file in dir_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                i = int(match.group(1))
                if i > max_i:
                    max_i = i
                    latest_file = file
    if latest_file is None:
        raise FileNotFoundError("No loop_{i}.pkl files found in directory")
    with open(latest_file, "rb") as f:
        data = pickle.load(f)
    return data, max_i, latest_file


def _append_metrics_to_json(path: str, entry: dict):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        else:
            data = []
    except Exception:
        data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _debug_candidate_stats(data: List[dict], tag: str, max_examples: int = 3) -> None:
    counts: List[int] = []
    for row in data:
        cands = row.get("candidates")
        if isinstance(cands, list):
            counts.append(len(cands))
        elif cands is None:
            counts.append(0)
        else:
            counts.append(-1)
    if not counts:
        print(f"[DEBUG] {tag}: no rows")
        return
    unique_counts = sorted(set(counts))
    unique_preview = unique_counts[:10]
    suffix = " ..." if len(unique_counts) > 10 else ""
    avg_count = sum(counts) / max(1, len(counts))
    print(
        f"[DEBUG] {tag}: rows={len(data)} "
        f"candidate_count[min/avg/max]={min(counts)}/{avg_count:.2f}/{max(counts)} "
        f"unique={unique_preview}{suffix}"
    )
    for idx, row in enumerate(data[:max_examples]):
        cands = row.get("candidates")
        cand_len = len(cands) if isinstance(cands, list) else 0
        first_type = type(cands[0]).__name__ if isinstance(cands, list) and cands else "None"
        nested_len = len(cands[0]) if isinstance(cands, list) and cands and isinstance(cands[0], list) else None
        print(
            f"[DEBUG] {tag}: row={idx} candidate_len={cand_len} "
            f"first_type={first_type} nested_len={nested_len}"
        )


def extract_question_from_prompt(prompt_cell: Any) -> str:
    return prompt_cell[0].get("content", "")


def extract_rg_solution(completion: str) -> Optional[str]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    parts = completion.split("</think>", 1)
    if len(parts) == 1:
        return None
    tail = parts[1].strip()
    final_response = tail.rsplit("Final Answer:", 1)
    if len(final_response) == 1:
        return None
    return final_response[1].strip()


def get_task_name(ds: Dataset) -> str:
    data_source = str(ds[0]['data_source'])
    data_source_lower = data_source.lower()
    if "arxiv_math" in data_source_lower or "arxivmath" in data_source_lower:
        return "arxiv_math"
    if "aime" in data_source_lower or "hmmt" in data_source_lower or "math" in data_source_lower or "deepscaler" in data_source_lower:
        return "math"
    elif "reasoning_gym" in data_source_lower:
        return "rg"
    elif "m-a-p/supergpqa" in data_source_lower:
        return 'supergpqa'
    elif data_source_lower == 'lcb':
        return 'code'
    else:
        raise ValueError(f"Unknown task: {data_source}")


# --------------------- prompt building ---------------------

def aggregate_prompt(question: str, candidate_answers: List[str], task: str) -> str:
    arxiv_loop_guidance = ""
    if task == 'rg':
        problem_kind = 'problem'
        format_hint = '<answer>...</answer>'
    elif task == 'supergpqa':
        problem_kind = 'multiple-choice problem'
        format_hint = '\\boxed{}. Only include the correct option letter in \\boxed{}; for example \\boxed{A}'
    elif task == 'arxiv_math':
        problem_kind = 'math problem'
        format_hint = '\\boxed{}'
        arxiv_loop_guidance = " Focus on producing the final answer and follow any additional formatting instructions in the question."
    else:
        problem_kind = 'math problem'
        format_hint = '\\boxed{}'

    parts = []
    if len(candidate_answers) == 1:
        parts.append(
            f"You are given a {problem_kind} and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Refine this trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. "
            "Show your reasoning step by step before giving the final answer. "
            f"End with the final result in {format_hint}."
            f"{arxiv_loop_guidance}\n"
        )
    else:
        parts.append(
            f"You are given a {problem_kind} and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. "
            "Aggregate the useful ideas and produce a single, high-quality solution. "
            "Reason carefully; if candidates disagree, choose the correct path. If all are incorrect, then attempt a different strategy. "
            "Show your reasoning step by step before giving the final answer. "
            f"End with the final result in {format_hint}."
            f"{arxiv_loop_guidance}\n"
        )

    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")

    if len(candidate_answers) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        ans_str = (candidate_answers[0] or "").strip()
        parts.append(f"---- Candidate ----\n{ans_str}\n")
        parts.append(
            f"Now refine the candidate into an improved solution. Show your reasoning step by step, then end with the final answer in {format_hint}."
        )
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, ans in enumerate(candidate_answers, 1):
            ans_str = (ans or "").strip()
            parts.append(f"---- Solution {i} ----\n{ans_str}\n")
        parts.append(
            f"Now write a single improved solution. Show your reasoning step by step, then end with the final answer in {format_hint}."
        )

    return "\n".join(parts)


def build_messages(question: str, candidate_answers: Optional[List[str]], task: str) -> List[Dict[str, str]]:
    """Build OpenAI-style chat messages for a single request."""
    if candidate_answers is not None:
        prompt = aggregate_prompt(question, candidate_answers, task)
    else:
        if task == 'arxiv_math':
            prompt = "\n".join(
                [
                    "You are given a difficult question. Your task is to solve the problem.",
                    "The question is written in such a way that it solely requires you to find the final answer. "
                    "Make sure to follow the additional formatting instructions if they are provided in the question.",
                    "Show your reasoning step by step, then put the final answer within \\boxed{}.",
                    "",
                    "Problem:",
                    question.strip(),
                ]
            )
        else:
            prompt = question
    return [{"role": "user", "content": prompt}]


def summarize_cot_prompt(question: str, candidate: str) -> str:
    parts = []
    parts.append(
        "You are given a math problem and a candidate solution. "
        "Summarize the solution into a concise chain-of-thought style outline that preserves all "
        "important information required to continue refinement later: the main approach(es), key steps/equations, "
        "useful intermediate results, and any mistakes or dead ends. "
        "Compress it while keeping the essential structure. "
        "If the candidate included a final answer, retain it at the end in \\boxed{ }.\n"
    )
    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solution:\n")
    parts.append(candidate.strip() + "\n")
    parts.append("Now produce the concise, information-preserving summary. "
                 "End with the final answer in \\boxed{} if present.")
    return "\n".join(parts)


def verify_cot_prompt(question: str, candidate: str) -> str:
    parts = []
    parts.append(
        "You are given a problem and a candidate solution. "
        "Verify whether the candidate solution is correct. "
        "If the solution is correct, output only True. "
        "If it is incorrect, output only False.  "
        "Do not generate anything else. "
    )
    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")
    parts.append("Candidate solution:\n")
    parts.append(candidate.strip() + "\n")
    parts.append("Now verify if the solution is True or False. Only output \"True\" or \"False\".")
    return "\n".join(parts)


# --------------------- async OpenAI batching ---------------------

async def _call_api(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    reasoning_effort: Optional[str],
    semaphore: asyncio.Semaphore,
) -> str:
    """Make a single OpenAI chat completion call with concurrency control."""
    max_retries = 3
    async with semaphore:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        # reasoning_effort is only supported by some models (e.g. o-series)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        # temperature / top_p may not be supported alongside reasoning_effort
        # on some models; let the user control this via args
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except openai.APIStatusError as e:
                if e.status_code == 503:
                    print(f"[_call_api] Attempt {attempt+1} got 503 Service Unavailable, retrying in 30s...")
                    await asyncio.sleep(30)
                elif attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[_call_api] Attempt {attempt+1} failed (HTTP {e.status_code}: {e}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[_call_api] Attempt {attempt+1} failed ({type(e).__name__}: {e}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    raise


async def batch_api_calls(
    client: AsyncOpenAI,
    model: str,
    all_messages: List[List[Dict[str, str]]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    reasoning_effort: Optional[str],
    max_concurrent: int,
) -> List[str]:
    """Run many chat completion calls concurrently with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        asyncio.ensure_future(
            _call_api(client, model, msgs, temperature, max_tokens, top_p, reasoning_effort, semaphore)
        )
        for msgs in all_messages
    ]
    # Show progress as tasks complete (arbitrary order)
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="API calls"):
        await fut
    # Gather preserves original order
    return [t.result() for t in tasks]


# --------------------- summarization ---------------------

async def summarize_candidates_inplace(
    client: AsyncOpenAI,
    model: str,
    data: List[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_effort: Optional[str],
    max_concurrent: int,
) -> None:
    all_messages = []
    idxs = []
    for pi, problem in enumerate(data):
        question = problem['orig_prompt']
        cands = problem.get('candidates') or []
        for ci, cand in enumerate(cands):
            prompt = summarize_cot_prompt(question, cand)
            all_messages.append([{"role": "user", "content": prompt}])
            idxs.append((pi, ci))

    if not all_messages:
        return

    responses = await batch_api_calls(
        client, model, all_messages, temperature, max_tokens, top_p, reasoning_effort, max_concurrent
    )
    for (pi, ci), summary in zip(idxs, responses):
        data[pi]['candidates'][ci] = summary


# --------------------- verification ---------------------

async def verify_candidates(
    client: AsyncOpenAI,
    model: str,
    data: List[dict],
    top_p: float,
    reasoning_effort: Optional[str],
    max_concurrent: int,
) -> List[int]:
    all_messages = []
    for problem in data:
        question = problem['orig_prompt']
        cands = problem.get('candidates') or []
        for cand in cands:
            prompt = verify_cot_prompt(question, cand)
            all_messages.append([{"role": "user", "content": prompt}])

    if not all_messages:
        return []

    responses = await batch_api_calls(
        client, model, all_messages, 0.1, 10, top_p, reasoning_effort, max_concurrent
    )
    print(responses[0])
    verified_vals = [
        1 if (m := re.findall(r'(true|false)', s, flags=re.I)) and m[-1].lower() == "true"
        else 0
        for s in responses
    ]
    return verified_vals


# --------------------- evaluation (unchanged) ---------------------

def evaluate_k_answers_math(k_answers: List[str], gt: str) -> Dict[str, Any]:
    solutions = [
        (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
        for a in k_answers
    ]
    extracted = [remove_boxed(s) for s in solutions]
    correct_bools = [bool(is_equiv(e, gt)) for e in extracted]
    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)

    clusters: List[Dict[str, Any]] = []
    for e in extracted:
        placed = False
        for c in clusters:
            if bool(is_equiv(e, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1})

    if not clusters:
        majority_vote = 0.0
    else:
        best = max(clusters, key=lambda c: c["count"])
        majority_vote = float(bool(is_equiv(best["rep"], gt)))

    return {
        "pred_accuracies": [float(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }


def evaluate_k_answers_arxiv_math(k_answers: List[str], gt: str) -> Dict[str, Any]:
    parsed_gt = parse_answer(str(gt))[0]
    parsed_answers: List[Any] = []
    correct_bools: List[bool] = []
    for model_reply in k_answers:
        parsed_reply = extract_answer(model_reply)[0]
        parsed_answers.append(parsed_reply)
        correct = check_answers(parsed_gt, parsed_reply)
        correct_bools.append(bool(correct))

    mean_acc = float(sum(correct_bools) / max(1, len(correct_bools)))
    pass_at_k = float(1.0 if any(correct_bools) else 0.0)

    clusters: List[Dict[str, Any]] = []
    for ans in parsed_answers:
        placed = False
        for c in clusters:
            if bool(check_answers(ans, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": ans, "count": 1})

    if not clusters:
        majority_vote = 0.0
    else:
        best = max(clusters, key=lambda c: c["count"])
        majority_vote = float(bool(check_answers(best["rep"], parsed_gt)))

    return {
        "pred_accuracies": [float(b) for b in correct_bools],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }


def evaluate_k_answers_rg(score_answer_fn: Callable[[str, str], float], k_answers: List[str], gt: dict) -> Dict[str, Any]:
    solutions = [extract_rg_solution(a) or "" for a in k_answers]
    scores = []
    for sol in solutions:
        try:
            scores.append(score_answer_fn(sol, gt))
        except:
            scores.append(0)
    mean_acc = float(sum(scores) / max(1, len(scores)))
    pass_at_k = float(1.0 if any(s == 1.0 for s in scores) else 0.0)

    clusters: List[Dict[str, Any]] = []
    for sol in solutions:
        placed = False
        for c in clusters:
            if bool(is_equiv(sol, c["rep"])):
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": sol, "count": 1})

    if not clusters:
        majority_vote = 0.0
    else:
        best = max(clusters, key=lambda c: c["count"])
        majority_vote = float(score_answer_fn(best["rep"], gt))

    return {
        "pred_accuracies": [float(s) for s in scores],
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }


# --------------------- main ---------------------

def generate_candidates(A, M, R):
    if A is None:
        return [None for _ in range(M)]
    return [random.sample(A, R) for _ in range(M)]


def reshape_list(lst, K):
    return [lst[i:i+K] for i in range(0, len(lst), K)]


async def run(
    client: AsyncOpenAI,
    model: str,
    k: int,
    population: int,
    data: List,
    task: str,
    self_verify: bool,
    temperature: float,
    max_tokens: int,
    top_p: float,
    reasoning_effort: Optional[str],
    max_concurrent: int,
    score_answer_fn: Optional[Callable[[str, str], float]] = None,
):
    all_messages, ground_truths, dataset_names = [], [], []
    print(f"[DEBUG] run start: problems={len(data)} k={k} population={population}")
    for problem in data:
        prompt = problem['orig_prompt']
        ground_truth = problem['gt']
        candidate_answers = generate_candidates(problem['candidates'], population, k)
        ground_truths.append(ground_truth)
        dataset_names.append(problem['dataset_name'])
        for candidates in candidate_answers:
            messages = build_messages(prompt, candidates, task)
            all_messages.append(messages)

    print(f"[DEBUG] Sending {len(all_messages)} API requests...")
    print(f"[DEBUG] First prompt: {all_messages[0][0]['content'][:200]}...")

    all_responses = await batch_api_calls(
        client, model, all_messages, temperature, max_tokens, top_p, reasoning_effort, max_concurrent
    )
    print(f"[DEBUG] First response: {all_responses[0][:200]}...")
    print(f"[DEBUG] run outputs: expected={len(data) * population} got={len(all_responses)}")

    response_lengths = [len(r.split()) for r in all_responses]  # word count as proxy
    median = np.percentile(response_lengths, 50)
    q25 = np.percentile(response_lengths, 25)
    q75 = np.percentile(response_lengths, 75)
    mean_response_length = sum(response_lengths) / max(1, len(response_lengths))

    all_responses = reshape_list(all_responses, population)

    for problem, responses in zip(data, all_responses):
        problem['candidates'] = responses
    _debug_candidate_stats(data, "after_run_assignment")

    if self_verify:
        verified_vals = await verify_candidates(
            client, model, data, top_p, reasoning_effort, max_concurrent
        )
        verified_vals = reshape_list(verified_vals, population)

    # Evaluate
    mean_acc: List[float] = []
    pass_at_k: List[float] = []
    majority_acc: List[float] = []
    verified_score_list: List[float] = []
    correct_bools = []

    for dataset_name, gt, responses in zip(dataset_names, ground_truths, all_responses):
        if task == 'rg':
            score_answer_fn = get_score_answer_fn(name=dataset_name)
            perf_metric = evaluate_k_answers_rg(score_answer_fn, responses[:], gt)
        elif task == 'arxiv_math':
            perf_metric = evaluate_k_answers_arxiv_math(responses[:], gt)
        else:
            perf_metric = evaluate_k_answers_math(responses[:], gt)
        mean_acc.append(perf_metric['mean_acc'])
        pass_at_k.append(perf_metric['pass_at_k'])
        majority_acc.append(perf_metric['majority_vote_correct'])
        correct_bools.append(perf_metric['pred_accuracies'])

    if self_verify:
        for dataset_name, gt, responses, verified in zip(dataset_names, ground_truths, all_responses, verified_vals):
            if task == 'rg':
                score_answer_fn = get_score_answer_fn(name=dataset_name)
                solutions = [extract_rg_solution(a) or "" for a in responses[:]]
                scores = [float(score_answer_fn(sol, gt)) for sol in solutions]
            else:
                solutions = [
                    (last_boxed_only_string(a) if last_boxed_only_string(a) is not None else "\\boxed{}")
                    for a in responses
                ]
                extracted = [remove_boxed(s) for s in solutions]
                scores = [float(is_equiv(e, gt)) for e in extracted]
            verified_score = sum([x*y for x,y in zip(scores, verified)]) / max(1, sum(verified))
            verified_score_list.append(verified_score)

    metrics = json.dumps(
        {
            "n_samples": len(mean_acc),
            "k": k,
            "mean_acc_k": sum(mean_acc) / max(1, len(mean_acc)),
            "mean_pass_at_k": sum(pass_at_k) / max(1, len(pass_at_k)),
            "mean_majority_acc": sum(majority_acc) / max(1, len(majority_acc)),
            "self_verified_acc": sum(verified_score_list) / max(1, len(verified_score_list)),
            "mean_length": mean_response_length,
            "median_length": median,
            "q25_length": q25,
            "q75_length": q75,
        }, indent=2
    )
    return data, metrics


async def loop(
    model: str,
    loops: int,
    k: int,
    population: int,
    summarize_cot: bool,
    seed_dataset: str,
    output_dir: str,
    max_new_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
    seed: int,
    resume: bool,
    remove_checkpoint: bool,
    reasoning_effort: Optional[str],
    self_verify: bool,
    base_url: Optional[str],
    api_key: Optional[str],
    max_concurrent: int,
):
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )

    score_answer_fn: Optional[Callable[[str, str], float]] = None

    if seed_dataset == 'arxiv_math':
        ds = load_dataset("MathArena/arxivmath", split="train")
    else:
        ds = Dataset.from_parquet(seed_dataset)

    task = "arxiv_math" if seed_dataset == "arxiv_math" else get_task_name(ds)

    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'k_'+str(k)+'_N_'+str(population)+'_seed_'+str(seed)+'.json')

    if not resume:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
    checkpoints_path = os.path.join(output_dir, 'checkpoints/' + 'k_'+str(k)+'_N_'+str(population)+'_seed_'+str(seed))
    os.makedirs(checkpoints_path, exist_ok=True)

    def build_initial_data() -> List[Dict[str, Any]]:
        if task == 'arxiv_math':
            return [
                {
                    'orig_prompt': row['problem'],
                    'dataset_name': 'arxiv_math',
                    'gt': row['answer'],
                    'candidates': None,
                }
                for row in ds
            ]
        return [
            {
                'orig_prompt': extract_question_from_prompt(row['prompt']),
                'dataset_name': (row['extra_info']['dataset_name'] if task == 'rg' else None),
                'gt': (json.loads(row['extra_info']['entry']) if task == 'rg' else row['reward_model']['ground_truth']),
                'candidates': None,
            }
            for row in ds
        ]

    if resume:
        try:
            data, start_loop_idx, _ = load_latest_loop_file(checkpoints_path)
            print(f'Starting Inference from Loop: {start_loop_idx + 1}')
            _debug_candidate_stats(data, "loaded_checkpoint")
        except:
            print(f'Checkpoint not found; defaulting to base')
            data = build_initial_data()
            start_loop_idx = -1
    else:
        data = build_initial_data()
        start_loop_idx = -1

    for loop_idx in range(start_loop_idx + 1, loops):
        data, metrics = await run(
            client=client,
            model=model,
            k=k,
            population=population,
            data=data,
            task=task,
            score_answer_fn=score_answer_fn,
            self_verify=self_verify,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            max_concurrent=max_concurrent,
        )
        checkpoint_file = os.path.join(checkpoints_path, f'loop_{loop_idx}.pkl')
        _debug_candidate_stats(data, f"pre_save_loop_{loop_idx}")
        with open(checkpoint_file, 'wb') as file:
            pickle.dump(data, file)
        try:
            with open(checkpoint_file, "rb") as file:
                reloaded_data = pickle.load(file)
            _debug_candidate_stats(reloaded_data, f"post_save_reload_loop_{loop_idx}")
        except Exception as e:
            print(f"[DEBUG] checkpoint reload failed at loop {loop_idx}: {e}")

        print(loop_idx, metrics)
        if summarize_cot and loop_idx < loops - 1:
            print("Summarizing candidates before aggregation...")
            await summarize_candidates_inplace(
                client=client,
                model=model,
                data=data,
                max_tokens=8192,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
                max_concurrent=max_concurrent,
            )
        metrics_dict = json.loads(metrics)

        out_entry = {
            "n_samples": metrics_dict.get("n_samples", None),
            "k": k,
            "population": population,
            "loop": loop_idx,
            "task": task,
            "mean_acc_k": metrics_dict["mean_acc_k"],
            "mean_pass_at_k": metrics_dict["mean_pass_at_k"],
            "mean_majority_acc": metrics_dict["mean_majority_acc"],
            "self_verified_acc": metrics_dict["self_verified_acc"],
            "mean_length": metrics_dict["mean_length"],
            "median_length": metrics_dict["median_length"],
            "q25_length": metrics_dict["q25_length"],
            "q75_length": metrics_dict["q75_length"],
        }

        _append_metrics_to_json(metrics_path, out_entry)
        print(f"Appended metrics for loop {loop_idx} to {metrics_path}")

    if remove_checkpoint:
        import shutil
        shutil.rmtree(checkpoints_path)


def main():
    ap = argparse.ArgumentParser(description="RSA eval loop using OpenAI-compatible API")
    ap.add_argument("--model", default="o4-mini",
                    help="Model name to use with the API")
    ap.add_argument("--base-url", default=None,
                    help="Custom base URL for OpenAI-compatible API (e.g. http://localhost:8000/v1)")
    ap.add_argument("--api-key", default=None,
                    help="API key (defaults to OPENAI_API_KEY env var)")
    ap.add_argument("--dataset", default="./data/aime25/train.parquet")
    ap.add_argument("--output", default="./eval")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--summarize-cot", action="store_true")
    ap.add_argument("--loops", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature (omit for reasoning models that don't support it)")
    ap.add_argument("--top-p", type=float, default=None,
                    help="Top-p / nucleus sampling (omit for reasoning models)")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="Reasoning effort for o-series models")
    ap.add_argument("--max-concurrent", type=int, default=32,
                    help="Max concurrent API requests")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action='store_true', default=False)
    ap.add_argument("--remove_checkpoint", action='store_true', default=False)
    ap.add_argument("--self_verify", action='store_true', default=False)
    args = ap.parse_args()

    asyncio.run(loop(
        model=args.model,
        loops=args.loops,
        seed_dataset=args.dataset,
        output_dir=os.path.join(args.output, args.model.split('/')[-1]),
        k=args.k,
        population=args.population,
        summarize_cot=args.summarize_cot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        resume=args.resume,
        remove_checkpoint=args.remove_checkpoint,
        self_verify=args.self_verify,
        reasoning_effort=args.reasoning_effort,
        base_url=args.base_url,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()
