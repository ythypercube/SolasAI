#!/usr/bin/env python3
"""
One-command improve pipeline:
1) build autofix dataset from logged failures
2) optionally merge autofix pairs into conversations.txt
3) run QA regression suite
4) optionally retrain model

Usage examples:
  /mnt/data/SolasAI/.venv/bin/python improve_pipeline.py
  /mnt/data/SolasAI/.venv/bin/python improve_pipeline.py --train-epochs 120
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], cwd: Path):
    print(f"\n$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-epochs', type=int, default=0)
    parser.add_argument('--skip-merge', action='store_true')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    py = sys.executable

    build_cmd = [py, 'build_feedback_dataset.py']
    if not args.skip_merge:
        build_cmd.append('--merge')
    run_step(build_cmd, script_dir)

    run_step([py, 'evaluate_qa.py'], script_dir)

    if args.train_epochs > 0:
        run_step([py, 'train.py', '--epochs', str(args.train_epochs)], script_dir)
        run_step([py, 'evaluate_qa.py'], script_dir)

    print('\nPipeline complete.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
