# Hisim Configuration Layout

Hisim keeps reusable platform and model facts in `src/hisim/spec`, while
`configs/simulation` contains concrete simulation scenarios.

- `src/hisim/spec/accelerator`: hardware registry used for KVCache capacity,
  HiCache bandwidth, and platform-level accounting.
- `src/hisim/spec/model`: built-in model registry used as an offline fallback.
  When SGLang provides a HuggingFace config at runtime, Hisim parses that config
  directly and does not rely on the built-in registry.
- `configs/simulation`: scenario files passed through `--sim-config-path` or
  `HISIM_CONFIG_PATH`. These select a platform, predictor database, and
  scheduler settings such as TP/EP/DP and data types.

AIConfigurator has its own system database. `predictor.device_name` must match a
system name available to that database, such as `h100_sxm` in the package
defaults or `h20_sxm` in the external Hisim AIC package. It is valid to use a
proxy predictor device for experiments, but the server will warn when it differs
from `platform.accelerator.name`.
