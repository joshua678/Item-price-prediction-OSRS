# Model Benchmarking

Generate test-split item price forecast plots from a trained model:

```bash
python -m benchmarking.plot_test_forecasts
```

Default behavior:

- Uses `prediction_model/model.pt` and `prediction_model/scaler.pkl`.
- Uses the same test split boundaries as training.
- Plots 288 input steps and 72 output steps.
- Uses 5-minute charting by default; `--plot-frequency 1h` renders hourly aggregated prices.
- Creates one plot per trained item by default.
- Plots the actual input/output prices, the model's p50 prediction points, plus shaded p10-p90 and p25-p75 prediction intervals at the model horizons.
- Writes PNGs plus a `manifest.csv` under `benchmarking/plots`.

Useful options:

```bash
python -m benchmarking.plot_test_forecasts --max-items 25 --plots-per-item 1 --plot-frequency 1h
python -m benchmarking.plot_test_forecasts --items "Old school bond,Abyssal whip"
python -m benchmarking.plot_test_forecasts --plot-frequency 1h
python -m benchmarking.plot_test_forecasts --plot-frequency 1h --hourly-aggregation last
python -m benchmarking.plot_test_forecasts --output-dir benchmarking/plots/latest
```

Hourly mode keeps model inference on the same 5-minute input windows, but renders actual prices in hourly buckets. Forecast intervals are drawn at model horizons that land on whole hours, such as 1h, 2h, 3h, and 6h for the default horizon list.

The checkpoint must have been trained with the current multi-horizon config. Older single-horizon checkpoints will fail fast with a horizon-count mismatch.
