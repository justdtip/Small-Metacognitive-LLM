import subprocess, sys
from pathlib import Path

def test_config_schema_and_harmony():
    p = subprocess.run([sys.executable, str(Path('tools/validate_configs.py'))], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout

