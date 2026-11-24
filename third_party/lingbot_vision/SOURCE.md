# Vendored source

- Upstream: https://github.com/robbyant/lingbot-vision
- Commit: `151e46321bae4399f8568829f190c7bdec216b49`
- License: Apache-2.0

Only the ViT-S/16 Python package and YAML config required by this repository are used. `loader.py` is limited to this variant, removes the legacy configuration merge and token-extraction APIs, and uses strict loading for complete backbone weights while retaining upstream checkpoint-wrapper unwrapping. `__init__.py` exports only the four loader entry points used here. Unused demos, preprocessing utilities and other model configs are excluded from the public repository.
