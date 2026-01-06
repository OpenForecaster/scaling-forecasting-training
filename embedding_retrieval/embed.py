"""Embedding construction, summarization, reranking, and passage chunking.

Small, focused helpers over vLLM/Transformers for:
- Building text inputs from raw records.
- Computing embedding vectors for documents and queries.
- Generating retrieval-optimized summaries.
- Reranking query-document pairs with a lightweight yes/no head.
"""

import math
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from functools import lru_cache
from typing import List, Tuple
import torch
from tqdm import tqdm

def construct_text(input_dict: dict, field: str) -> tuple[list[str], list[str]]:
    """Build text strings for embedding from `input_dict`.

    Concatenates optional title and description with the requested field.
    Supports fields: 'maintext', 'summary', or 'question_title'.
    Returns parallel lists of keys and text strings.
    """
    allowed_fields = {"maintext", "summary", "question_title"}
    if field not in allowed_fields:
        raise ValueError(f"field must be one of {sorted(allowed_fields)}")

    texts: list[str] = []
    keys: list[str] = []
    for key, value in input_dict.items():
        content = value.get(field)
        if not isinstance(content, str) or not content.strip():
            continue

        title = value.get("title", "")
        if not isinstance(title, str):
            title = ""
        description = value.get("description", "")
        if not isinstance(description, str):
            description = ""
            
        background = value.get("background", "")
        if not isinstance(background, str):
            background = ""
            
        resolution = value.get("resolution_criteria", "")
        if not isinstance(resolution, str):
            resolution = ""

        parts = [title, description, content]
        if "question" in field:
            parts.extend([background, resolution])
        text = "\n".join(p for p in parts if p).replace("\n\n", "\n")
        texts.append(text)
        keys.append(key)

    assert len(texts) == len(keys), "Mismatch between texts and keys length"
    print(f"{len(texts)} texts to embed from '{field}'")
    return keys, texts


def get_embeddings(text: list[str], is_query: bool, model_name: str = 'Qwen/Qwen3-Embedding-8B') -> np.ndarray:
    """Compute embedding vectors for `text` with an embedding LLM.

    Adds a task-specific instruction prefix to better steer embeddings for
    queries vs. documents. Returns a NumPy array of dtype float32.
    """
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'
    
    task = "Given a question, produce a retrieval-optimized embedding that captures the information need (intent, key entities, and constraints) so the most relevant answer-bearing passages can be found." if is_query else "Given a document, produce a retrieval-optimized embedding that captures the key entities, facts, and relationships so it can be matched to relevant queries."
    text = [get_detailed_instruct(task, t) for t in text]
    text = [t[:40900] for t in text]  # Truncate to fit model context
    
    model = LLM(model=model_name, task="embed")
    outputs = model.embed(text)
    embeddings = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)

    return embeddings


import re
def _strip_thinking(output_text: str) -> str:
        patterns = [
            (r"<think>.*?</think>", re.DOTALL),
            (r"<\|thinking\|>.*?<\|/thinking\|>", re.DOTALL),
            (r"<think_start>.*?<think_end>", re.DOTALL),
            (r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL),
        ]
        s = output_text
        for pat, flags in patterns:
            s = re.sub(pat, " ", s, flags=flags)
        return re.sub(r"\s+", " ", s).strip()

def get_text(input) -> str:
    if not input.outputs:
        raise ValueError("No output from model")
    return input.outputs[0].text

def get_summary(text: list[str], model_name: str = "qwen/Qwen3-14B", max_new_tokens: int = 20480, temperature: float = 0.6, top_p: float = 0.9, batch_size: int = 1) -> list[str]:
    system_prompt = (
        "You are an expert at writing dense, retrieval-oriented summaries (for RAG). "
        "Produce a compact paragraph that preserves all factual content needed for search: "
        "named entities, dates/times, numbers, places, definitions, causal/temporal relations, and key outcomes. "
        "Remove rhetoric and filler. Normalize names and abbreviations. Do not speculate. "
        "Think internally but output only the final compressed summary."
    )


    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.90, dtype="auto", max_model_len=24576)

    user_prompt = ["Summarize the following article into a single compact paragraph optimized for retrieval. Retain all facts and relationships; omit fluff and repetition.\n\n" + t for t in text]
    prompts = ["System: " + system_prompt + "\nUser: " + u + "\nAssistant:" for u in user_prompt]

    summaries: list[str] = []
    bsz = batch_size
    for i in range(0, len(prompts), bsz):
        batch_prompts = prompts[i:i+bsz]
        outs = llm.generate(batch_prompts, sampling_params)
        out_text = list(map(get_text, outs))
        summaries.extend(out_text)

    summaries = list(map(_strip_thinking, summaries))
    assert len(summaries) == len(text), "Mismatch between summaries and input text length"
    return summaries

def format_instruction(instruction: str, query: str, doc: str):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(tokenizer, pairs, instruction, max_length, suffix_tokens):
    from vllm.inputs.data import TokensPrompt
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ids[:max_length] + suffix_tokens for ids in messages]
    messages = [TokensPrompt(prompt_token_ids=ids) for ids in messages]
    return messages

def compute_logits(model, messages, sampling_params, true_token, false_token):
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores


def rerank_docs(model_name: str, query_doc_pairs: list[tuple[str, str]]) -> list[float]:
    """Score (query, doc) pairs with a constrained yes/no head.

    Returns a list of scores in [0, 1] where higher means more relevant.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    max_length = 32000
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    model = LLM(model=model_name, max_model_len=32000, enable_prefix_caching=True, gpu_memory_utilization=0.95)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )
    task = 'Given a question, rank the articles by the most relevant answer-bearing passages.'

    inputs = process_inputs(tokenizer, query_doc_pairs, task, max_length - len(suffix_tokens), suffix_tokens)
    scores = compute_logits(model, inputs, sampling_params, true_token, false_token)
    return scores

@lru_cache(maxsize=2)
def _get_tokenizer_cached(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

def chunk_texts(
    texts: List[str],
    model_name: str,
    max_tokens: int = 512,
    stride: int = 64,
) -> Tuple[List[str], List[int]]:
    """Tokenize and split texts into overlapping chunks by tokens.
    Returns a tuple `(chunks, chunk_to_doc_idx)` where `chunks[i]` is the text of
    the i-th chunk and `chunk_to_doc_idx[i]` is the source document index.
    """
    tok = _get_tokenizer_cached(model_name)
    chunks: List[str] = []
    mapping: List[int] = []
    for doc_idx, t in tqdm(enumerate(texts), total=len(texts), desc="Chunking texts"):
        if not t:
            continue
        ids = tok(t, add_special_tokens=False).input_ids
        n = len(ids)
        if n == 0:
            continue
        if n <= max_tokens:
            chunks.append(t)
            mapping.append(doc_idx)
            continue
        start = 0
        step = max(max_tokens - stride, 1)
        while start < n:
            end = min(start + max_tokens, n)
            piece = tok.decode(ids[start:end], skip_special_tokens=True)
            if piece.strip():
                chunks.append(piece)
                mapping.append(doc_idx)
            if end == n:
                break
            start += step
    return chunks, mapping
