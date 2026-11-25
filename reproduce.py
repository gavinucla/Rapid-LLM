"""Reproduce all Project 2 results in one go.

    python reproduce.py

Runs every case study (about 80 RAPID-LLM runs; about 25 minutes), the slide 7
breakdown, and the plots. Results end up in output/project2/. See README.md.
"""

import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parent
for script in ("project2/run_studies.py", "project2/breakdown.py", "project2/make_plots.py"):
    print(f"\n=== {script}", flush=True)
    subprocess.run([sys.executable, str(repo / script)], check=True, cwd=repo)

print("\nDone. Results are in output/project2/ (README.md says which output goes with which slide).")
