# Modal Sandbox Backend

FeatureBench can run inference (`fb infer`) and evaluation (`fb eval`) on
[Modal](https://modal.com) Sandboxes instead of a local Docker daemon. Each task
runs in a `modal.Sandbox` created from the task's benchmark image, so the images
run natively on `linux/amd64` and on real GPUs — without a local Docker daemon.

This is opt-in via `--backend modal` and fully **accretive**: the default
`--backend docker` path is unchanged, and `modal` is imported lazily so the
Docker path never requires it.

## Why

The default backend talks to a local Docker daemon. On machines without a
suitable Docker setup (e.g. Apple Silicon, where benchmark `amd64` images run
under slow emulation and GPUs are unavailable), the Modal backend lets you:

- run `amd64` images natively (no emulation), and
- run GPU tasks on real GPUs (e.g. A100), which a local non-NVIDIA host cannot.

## Prerequisites

1. **Install deps** (modal is already declared in `pyproject.toml`):
   ```bash
   uv sync
   ```
2. **Authenticate Modal** (creates `~/.modal.toml`):
   ```bash
   uv run modal setup
   ```
3. **Create `config.toml`** in the repo root (gitignored). Copy from
   `config_example.toml`. For agent credentials you can also pass
   `--api-key` / `--base-url` on the command line, which override the config.

## Inference on Modal

Add `--backend modal` to any `fb infer` invocation:

```bash
uv run fb infer \
  --agent mini_swe_agent \
  --model openrouter/z-ai/glm-5.1 \
  --api-key "$OPENROUTER_API_KEY" \
  --split lite \
  --task-id pydantic__pydantic.e1dcaf9e.test_deprecated_fields.40a2ec54.lv1 \
  --n-concurrent 1 \
  --backend modal
```

The chosen backend is recorded in `run_metadata.json`, so `--resume` reuses it.

> OpenRouter note: `mini_swe_agent` already supports the `openrouter` provider.
> Use a `openrouter/<model>` model id and supply the key via `--api-key`
> (mapped to `MSWEA_API_KEY` / `OPENROUTER_API_KEY`); no base URL is required.

## Evaluation on Modal

Add `--backend modal` to any `fb eval` invocation:

```bash
# Evaluate a predictions file on Modal
uv run fb eval -p runs/<timestamp>/output.jsonl --split lite --backend modal --n-concurrent 6

# Verify gold/oracle patches on Modal (lv1 only; gold mode skips lv2)
uv run fb eval -p gold --split lite --backend modal --n-concurrent 6
```

The eval backend reuses the real harness logic (`run_instance_level1` /
`run_instance_level2`) verbatim — only the container is swapped for a Modal
sandbox — so results are identical in meaning to the Docker backend.

## GPU mapping

GPU requirements come from each task's `repo_settings`
(`docker_runtime_config.need_gpu` and `number_once`):

- `need_gpu = false` → CPU sandbox.
- `need_gpu = true`, `number_once = 1` → `gpu="A100"`.
- `need_gpu = true`, `number_once = N` → `gpu="A100:N"` (multi-GPU).

Override the GPU type via the `FB_MODAL_GPU` environment variable
(e.g. `FB_MODAL_GPU=H100`, `FB_MODAL_GPU=L40S`). On the Modal backend the local
host-GPU scheduler / `--gpu-ids` are bypassed (the sandbox provides the GPU).

## Image pre-warming (recommended for batches)

The first time an image is used on Modal it is pulled into Modal's registry
(~5 min for the large benchmark images); subsequent sandboxes start in
seconds. To avoid paying GPU time during that one-time pull, pre-warm all
distinct images on cheap CPU sandboxes first. Minimal example:

```python
import modal
from datasets import load_dataset

app = modal.App.lookup("featurebench-infer", create_if_missing=True)
images = sorted({r["image_name"] for r in load_dataset("LiberCoders/FeatureBench", split="lite")})
for img in images:
    sb = modal.Sandbox.create(image=modal.Image.from_registry(img, add_python=None), app=app, timeout=900)
    sb.exec("bash", "-c", "true").wait()
    sb.terminate()
```

## Architecture

- `featurebench/infer/modal_container.py` — `ModalContainerManager`, a drop-in
  for `infer/container.py:ContainerManager` (same public methods:
  `pull_image`, `create_container`, `exec_command`, `exec_command_stream`,
  `copy_to_container`, `copy_from_container`, `stop_container`).
- `featurebench/harness/modal_container.py` — `ModalEvalContainerManager` +
  `ModalEvalContainer`, a drop-in for `harness/container.py:EvalContainerManager`.
  The container shim implements the docker-py surface the harness touches
  (`exec_run`, `put_archive`, `kill`/`remove`); sandbox lifecycle is delegated
  to the infer `ModalContainerManager`.
- Both `fb infer` and `fb eval` select the backend via `--backend {docker,modal}`
  (default `docker`). A lazy factory keeps `modal` out of the Docker path.

Modal specifics used (modal ≥ 1.5): `modal.App.lookup(name, create_if_missing=True)`,
`modal.Image.from_registry(image, add_python=None)`,
`modal.Sandbox.create(image=, app=, timeout=, workdir=, gpu=, secrets=[Secret.from_dict(env)])`,
`sandbox.exec(*cmd)` → `.stdout/.stderr/.wait()/.returncode`, and
`sandbox.filesystem.write_bytes(data, path)` / `read_bytes(path)`.

## Notes & limitations

- Environment variables are injected via `modal.Secret.from_dict(...)`. The
  Docker-only `localhost → 172.17.0.1` host-gateway rewrite and `--proxy-port`
  flow are not applied on Modal (public endpoints such as OpenRouter work as-is).
- The host cache mount (`/download`) is not used on Modal; agents reinstall per
  sandbox. Leave `[infer].download_cache_dir` empty.
- First-use image pulls are slow (~5 min/image, one-time); A100 provisioning can
  occasionally queue for a few minutes under concurrency. Neither is a failure.
- `fb eval -p gold` only covers tasks that ship a gold `patch`. In the current
  `LiberCoders/FeatureBench` lite snapshot that is the 26 lv1 tasks; the 4 lv2
  tasks carry no gold patch (their hidden tests `import agent_code`, the agent's
  from-scratch solution), so they are excluded from gold mode by design.
