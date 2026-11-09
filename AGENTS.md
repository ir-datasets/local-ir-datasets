# Agent Instructions

## Development environment

Work inside the dev container configured in `.devcontainer/` (Python 3.11 plus
`ir_datasets` from the `v2` branch). Do not create or use local Python virtual
environments. Build the image and run commands in it, e.g.:

```bash
docker build -t local-ir-datasets-dev .devcontainer
docker run --rm -v "$PWD":/workspaces/local-ir-datasets \
  -w /workspaces/local-ir-datasets local-ir-datasets-dev \
  bash -c "pip install --no-deps -e . && pytest"
```

## Good engineering practices

- Keep changes small, focused, and consistent with existing code style.
- Add or update tests for any behavior change; run `pytest` before finishing.
- Only add comments where intent is not obvious.
- Update the README when behavior or setup changes.
- Do not commit secrets or sensitive data.
