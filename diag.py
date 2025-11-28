# diagnostic_inspect_app_config.py
from dataclasses import fields
from src.config import APP_CONFIG
print("Top-level fields on APP_CONFIG:", [f.name for f in fields(APP_CONFIG.__class__)])
print("Model fields:", [f.name for f in fields(APP_CONFIG.model.__class__)])
print("Training fields:", [f.name for f in fields(APP_CONFIG.training.__class__)])
print("Data fields:", [f.name for f in fields(APP_CONFIG.data.__class__)])
# evaluation may be named 'evaluation' or 'eval' — list its fields if present
if hasattr(APP_CONFIG, "evaluation"):
    print("Evaluation fields:", [f.name for f in fields(APP_CONFIG.evaluation.__class__)])
elif hasattr(APP_CONFIG, "eval"):
    print("Evaluation fields:", [f.name for f in fields(APP_CONFIG.eval.__class__)])
else:
    raise RuntimeError("APP_CONFIG has no 'evaluation' or 'eval' attribute; check src.config AppConfig.")

