# COMP4320: Machine Learning III - Project NexUS
**Group:** Malhar Gohel, Min Sithu, Chris Botoman

## Overview
This repository contains a federated multi-agent reinforcement learning prototype for traffic signal optimization.
The training script now supports SUMO environment interaction for local agent updates and Federated Averaging across agents.

## Setup
1. Open PowerShell in the project root.
2. Activate the virtual environment:
   - `.\\test.venv\Scripts\Activate.ps1`
3. Install dependencies if needed:
   - `python -m pip install -r requirements.txt`

## Run training
From the project root:

```powershell
python train.py --use-env
```

For a faster test run without SUMO:

```powershell
python train.py --rounds 10 --local-epochs 2
```

## Command options
- `--use-env`: enable SUMO-based local training when `net_file` and `route_file` are available
- `--rounds N`: number of federated communication rounds
- `--local-epochs N`: local episodes per agent before aggregation
- `--output-dir DIR`: directory for saved outputs

## Where to see results
The script saves output in the `results` folder:
- `results/training_results.png`
- `results/training_history.json`
- `results/global_model.pth`

## Configuration
The default hyperparameters are in `configs/params.yaml`.
Update `simulation.net_file` and `simulation.route_file` to point to your SUMO network and route files.

## SUMO setup
If you want real SUMO training, set the `SUMO_HOME` environment variable and install SUMO.
The code will fallback to placeholder training when SUMO is unavailable.

## Notes
- If SUMO map files are missing, the code will fall back to placeholder local training.
- For actual traffic-signal training, provide valid SUMO files in `configs/params.yaml` and run with `--use-env`.
