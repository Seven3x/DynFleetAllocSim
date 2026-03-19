#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m milp_sim.main --gui-online
