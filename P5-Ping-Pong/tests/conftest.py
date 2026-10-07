"""Hermetic test runs: measured bench numbers in config_local.json must not change a test."""

import os

os.environ["PP_NO_LOCAL"] = "1"
