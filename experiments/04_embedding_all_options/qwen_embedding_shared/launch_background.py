#!/usr/bin/env python3
"""Preflight then detach the two-stage runner, waiting for existing GPU work."""
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import sys

import run_retrieval


def main():
    forwarded = list(sys.argv[1:])
    args = run_retrieval.resolve_args(run_retrieval.parser().parse_args(forwarded))
    script = Path(__file__).resolve().with_name('run_retrieval.py')
    command = [sys.executable, '-u', str(script), *forwarded]
    preflight = [sys.executable, '-u', str(script),
                 *[arg for arg in forwarded if arg not in {'--check-only', '--score-only'}], '--check-only']
    if not args.score_only:
        subprocess.run(preflight, check=True)
    if args.check_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'A retrieval job is already running for {args.out}')
    if '--wait-for-idle' not in command and not args.score_only:
        command.append('--wait-for-idle')
    logs = args.root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log = logs / f'qwen_embedding06_shared_{stamp}.log'
    with log.open('wb') as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    latest = logs / 'qwen_embedding06_shared_latest.log'
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(log.name)
    print(f'Retrieval queue started: PID {process.pid}. You may close this terminal.')
    print('It waits for the existing comparison/training job and GPU to become idle.')
    print(f'Log: {log}')
    print(f'Status: {args.out / "status.json"}')
    print(f'Monitor: tail -f {latest}')


if __name__ == '__main__':
    main()
