import subprocess, sys

def test_id_harmony():
    p = subprocess.run([sys.executable, 'tools/assert_config_ids.py', '--base-dir', 'model/Base'], capture_output=True, text=True)
    assert p.returncode == 0, f"stdout={p.stdout}\nstderr={p.stderr}"

