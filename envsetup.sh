#!/bin/bash

# Script to run in OpenAI codex environment. This
# downloads everything that we need to get before
# internet access is turned off.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run uvbootstrapper.py
