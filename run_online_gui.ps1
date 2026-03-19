$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"
python -m milp_sim.main --gui-online
