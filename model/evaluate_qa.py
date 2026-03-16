#!/usr/bin/env python3
"""
Lightweight QA regression evaluator for SolasAI.
Runs questions through answer_message() and checks expected patterns.

Usage:
  /mnt/data/SolasAI/.venv/bin/python evaluate_qa.py
"""

import re
from dataclasses import dataclass

import inference_server as server


@dataclass
class Case:
    question: str
    expected_pattern: str


def format_number_for_regex(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        integer = int(round(value))
        with_commas = f"{integer:,}"
        return re.escape(with_commas).replace(',', ',?')
    as_text = f"{value:,.2f}".rstrip('0').rstrip('.')
    return re.escape(as_text).replace(',', ',?')


def build_cases() -> list[Case]:
    unit_seconds = {
        'second': 1,
        'minute': 60,
        'hour': 3600,
        'day': 86400,
        'week': 604800,
        'year': 31536000,
    }

    basic_cases = [
        Case('hello', r'hello|hi'),
        Case('what is the capital of france', r'paris'),
        Case('which planet is known as the red planet', r'mars'),
        Case('what is h2o', r'water'),
        Case('what is 10 times 12', r'120'),
        Case('what are vectors', r'vectors|numbers|meaning'),
        Case('what is scratch', r'scratch'),
        Case('what is turbowarp', r'turbowarp'),
        Case('how many years until 2029?', r'years until 2029|2029 is this year|2029 was'),
    ]

    unit_pairs = [
        ('second', 'minute'), ('second', 'hour'), ('second', 'day'), ('second', 'week'),
        ('minute', 'hour'), ('minute', 'day'), ('minute', 'week'),
        ('hour', 'day'), ('hour', 'week'), ('hour', 'year'),
        ('day', 'week'), ('day', 'year'),
        ('week', 'year'),
    ]

    article_variants = ['a', 'an', 'one']
    punctuation_variants = ['', '?']

    generated_cases: list[Case] = []
    for asked, container in unit_pairs:
        expected_ratio = unit_seconds[container] / unit_seconds[asked]
        pattern = format_number_for_regex(expected_ratio)
        for article in article_variants:
            for punctuation in punctuation_variants:
                q1 = f'How many {asked}s are in {article} {container}{punctuation}'
                q2 = f'how many {asked}s are in {article} {container}{punctuation}'
                generated_cases.append(Case(q1, pattern))
                generated_cases.append(Case(q2, pattern))

    quantity_cases: list[Case] = []
    numeric_templates = [
        ('seconds', 2, 'hour'),
        ('minutes', 3, 'day'),
        ('hours', 2, 'week'),
        ('days', 2, 'year'),
        ('minutes', 1.5, 'hour'),
        ('seconds', 0.5, 'minute'),
        ('hours', 24, 'day'),
    ]
    for asked_plural, quantity, container in numeric_templates:
        asked_singular = asked_plural.rstrip('s')
        expected_ratio = (quantity * unit_seconds[container]) / unit_seconds[asked_singular]
        pattern = format_number_for_regex(expected_ratio)
        q = f'How many {asked_plural} are in {quantity:g} {container}?' 
        quantity_cases.append(Case(q, pattern))

    all_cases = basic_cases + generated_cases + quantity_cases
    return all_cases


def run_case(case: Case) -> tuple[bool, str]:
    reply = server.answer_message(case.question, [])
    ok = re.search(case.expected_pattern, reply, flags=re.IGNORECASE) is not None
    return ok, reply


def main() -> int:
    server.knowledge_pairs = server.load_knowledge_pairs()
    server.build_vector_index(server.knowledge_pairs)
    server.build_embedding_index(server.knowledge_pairs)

    cases = build_cases()

    passed = 0
    failed = []

    for case in cases:
        ok, reply = run_case(case)
        if ok:
            passed += 1
        else:
            failed.append((case, reply))

    total = len(cases)
    print(f'Passed {passed}/{total} cases')

    if failed:
        print('\nFailed cases:')
        for case, reply in failed:
            print(f'- Q: {case.question}')
            print(f'  expected: /{case.expected_pattern}/i')
            print(f'  got: {reply}')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
