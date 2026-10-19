import subprocess
import sys


def test_serving_does_not_import_pytorch_or_sklearn():
    """The inference image ships without PyTorch: importing the API must not pull it in (sklearn may appear only because LightGBM imports it when installed)."""
    code = "import sys; import inbox.serving.app; bad = [m for m in ('torch', 'onnxscript') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
