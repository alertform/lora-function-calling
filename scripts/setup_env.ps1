# One-shot env setup on the RTX 4060 box (Windows PowerShell).
# The system Python is 3.13 + CPU torch — too new for the CUDA ML stack, so we
# use `uv` to stand up an isolated Python 3.11 venv with a CUDA build of torch.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
$ErrorActionPreference = "Stop"

# 1. uv (fast Python/venv/deps manager) — installs to ~/.local/bin if missing
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "installing uv ..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# 2. Python 3.11 venv
uv python install 3.11
uv venv --python 3.11 .venv
$py = ".\.venv\Scripts\python.exe"

# 3. CUDA torch FIRST (cu124 wheels match driver 610.x / CUDA 12.x)
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. the rest
& $py -m pip install -r requirements.txt

# 5. sanity
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
& $py -c "import bitsandbytes as bnb; print('bitsandbytes', bnb.__version__)"

Write-Host "`n✅ env ready. Next:"
Write-Host "   .\.venv\Scripts\python.exe src\prepare_data.py --dataset Salesforce/xlam-function-calling-60k"
Write-Host "   .\.venv\Scripts\python.exe src\train.py   --config config\qlora.yaml"
Write-Host "   .\.venv\Scripts\python.exe src\eval.py    --config config\qlora.yaml"
