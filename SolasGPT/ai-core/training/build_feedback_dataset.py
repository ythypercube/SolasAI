#!/usr/bin/env python3
"""
Convert logged low-quality chat events into auto-fix training examples.

Input: data/feedback_log.jsonl
Output: data/feedback_autofix.txt (User/Assistant pairs)
Optional: merge examples into data/conversations.txt

Usage:
  /mnt/data/SolasAI/.venv/bin/python build_feedback_dataset.py
  /mnt/data/SolasAI/.venv/bin/python build_feedback_dataset.py --merge
"""

import argparse
import json
import os
from typing import Iterable

import inference_server as server


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_LOG = os.path.join(BASE_DIR, 'data', 'feedback_log.jsonl')
AUTOFIX_DATA = os.path.join(BASE_DIR, 'data', 'feedback_autofix.txt')
MAIN_DATA = os.path.join(BASE_DIR, 'data', 'conversations.txt')


def load_feedback_events(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    events = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def parse_existing_pairs(path: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    pairs: set[tuple[str, str]] = set()
    pending_user = None
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith('User: '):
                pending_user = line[6:].strip()
            elif line.startswith('Assistant: ') and pending_user:
                pairs.add((pending_user, line[11:].strip()))
                pending_user = None
    return pairs


def improved_answer(question: str) -> str | None:
    question = str(question or '').strip()
    if not question:
        return None

    direct = server.factual_reply(question)
    if direct:
        return server.clean_reply(direct)

    math_value = server.evaluate_math(question)
    if math_value is not None:
        return server.clean_reply(f'The answer is {math_value}.')

    retrieval, score = server.retrieval_reply(question, [])
    if retrieval and score >= 0.80:
        return server.clean_reply(retrieval)

    return None


def iter_autofix_pairs(events: Iterable[dict]) -> Iterable[tuple[str, str]]:
    for event in events:
        if not event.get('needsImprovement'):
            continue
        question = str(event.get('user', '')).strip()
        answer = improved_answer(question)
        if not answer:
            continue
        if answer.lower().startswith('i am still learning'):
            continue
        if 'ask a specific question' in answer.lower():
            continue
        yield question, answer


def write_pairs(path: str, pairs: list[tuple[str, str]]):
    with open(path, 'w', encoding='utf-8') as handle:
        for question, answer in pairs:
            handle.write(f'User: {question}\n')
            handle.write(f'Assistant: {answer}\n')


def append_unique_pairs(path: str, pairs: list[tuple[str, str]]):
    existing = parse_existing_pairs(path)
    to_add = [(q, a) for q, a in pairs if (q, a) not in existing]
    if not to_add:
        return 0

    with open(path, 'a', encoding='utf-8') as handle:
        for question, answer in to_add:
            handle.write(f'\nUser: {question}\n')
            handle.write(f'Assistant: {answer}\n')
    return len(to_add)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback-log', type=str, default=FEEDBACK_LOG)
    parser.add_argument('--output', type=str, default=AUTOFIX_DATA)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--merge-target', type=str, default=MAIN_DATA)
    args = parser.parse_args()

    server.knowledge_pairs = server.load_knowledge_pairs()
    server.build_vector_index(server.knowledge_pairs)
    server.build_embedding_index(server.knowledge_pairs)

    events = load_feedback_events(args.feedback_log)
    pairs = list(dict.fromkeys(iter_autofix_pairs(events)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_pairs(args.output, pairs)
    print(f'Autofix pairs written: {len(pairs)} -> {args.output}')

    if args.merge:
        added = append_unique_pairs(args.merge_target, pairs)
        print(f'Merged unique pairs: {added} -> {args.merge_target}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
