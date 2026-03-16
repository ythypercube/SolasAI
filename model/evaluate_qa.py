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


def run_case(case: Case) -> tuple[bool, str]:
    reply = server.answer_message(case.question, [])
    ok = re.search(case.expected_pattern, reply, flags=re.IGNORECASE) is not None
    return ok, reply


def main() -> int:
    server.knowledge_pairs = server.load_knowledge_pairs()
    server.build_vector_index(server.knowledge_pairs)
    server.build_embedding_index(server.knowledge_pairs)

    cases = [
        Case('hello', r'hello|hi'),
        Case('How many days are in a week?', r'7'),
        Case('How many hours are in a day?', r'24'),
        Case('How many minutes are in an hour?', r'60'),
        Case('How many seconds are in an hour?', r'3,?600'),
        Case('How many seconds are in a day?', r'86,?400'),
        Case('how many years until 2029?', r'years until 2029|2029 is this year|2029 was'),
        Case('what is the capital of france', r'paris'),
        Case('which planet is known as the red planet', r'mars'),
        Case('what is h2o', r'water'),
        Case('what is 10 times 12', r'120'),
        Case('what are vectors', r'vectors|numbers|meaning'),
        Case('what is scratch', r'scratch'),
        Case('what is turbowarp', r'turbowarp'),
    ]

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
