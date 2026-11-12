# Pump.fun top-five forecaster

An offline, deterministic Python forecaster for `pumpfun-top5-forecast-01`.

`main.py` reads `task.json` and `history.ndjson` from `MAKEFUN_INPUT_DIR` (default `/input`) and writes the contract's `forecast.json` under `MAKEFUN_OUTPUT_DIR` (default `/output`). It uses only the Python standard library.

The forecast combines repeatable stochastic cap paths for coins visible in the history with a probability reserve for launches that happen after the cut. It aggregates rank and size distributions across the task's forecast reads. The simulation seed is derived from the last observed mark, so identical input bytes produce identical output bytes.

For a local smoke run from this directory:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
