# jbr-fed-dp-training

Differential privacy training utilities for PyTorch and HuggingFace.

## What is this package?

Provides tools for training models with differential privacy guarantees, including:
- DP-SGD trainers for HuggingFace Transformers
- Privacy accounting and argument management
- PyTorch training loop implementations with DP
- Patched components for privacy-preserving training

This is a reusable Python package inside the monorepo, located at `packages/dp-training`.
- Distribution name (for installation): `jbr-fed-dp-training` (from `pyproject.toml` → `[project].name`).
- Folder slug (used in tags/CI paths): `dp-training`.

Packages are intended to be shared across workspaces and other packages. They are versioned and released independently via Git tags.

## Layout

```
packages/dp-training/
├─ pyproject.toml
├─ README.md
├─ src/
│  └─ jbr/           # namespace per PEP 420
└─ tests/            # optional
```

## Development (local)

- Sync dependencies:
  ```bash
  uv sync
  ```
- Run tests (if present):
  ```bash
  uv run pytest -q
  ```

This template pins Python ≥3.11 and uses uv. The `jbr-fed-core` dependency is linked as an editable sibling via `[tool.uv.sources]` during development.

## Using this package (install)

From your environment (e.g., a workspace) you can install the released distribution from your configured Python package repository:

```bash
# Example using a private Artifact Registry URL exposed as $PYTHON_REPOSITORY_URL
uv pip install --extra-index-url "$PYTHON_REPOSITORY_URL" jbr-fed-dp-training
```

During monorepo development, local editable installs are configured through `[tool.uv.sources]` in `pyproject.toml`.

## CI behavior (monorepo)

The repository’s central CI discovers every folder under `packages/` and runs tests for those that contain a `tests/` directory. No CI edits are required when adding a new package.

Relevant paths for this package:
- Sources and tests: `packages/dp-training/**`
- CI workflow: `.github/workflows/ci.yaml`

## Releases and versioning

- Versioning is derived from Git tags (`0.0.0` set in `pyproject.toml` for local builds).
- Tag format: `dp-training-vX.Y.Z` (e.g., `dp-training-v0.1.0`).

Release flow:
```bash
# Create and push a release tag for this package
git tag dp-training-v0.1.0
git push origin dp-training-v0.1.0
```
This triggers the `Release Package` workflow which builds and publishes only this package. 
If this package depends on internal packages, ensure compatible versions are available (release dependencies first when needed).

## License

Licensed under the Apache License, Version 2.0. See the repository LICENSE file and https://www.apache.org/licenses/LICENSE-2.0.
