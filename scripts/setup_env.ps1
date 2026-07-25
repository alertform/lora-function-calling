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
#    NOTE: a uv venv ships no pip — install through `uv pip`, not `python -m pip`.
uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu124

# 4. the rest
uv pip install --python $py -r requirements.txt

# 5. sanity — must actually pass; a native exe's non-zero exit does NOT trip
#    $ErrorActionPreference, so gate the success banner on $LASTEXITCODE.
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { Write-Error "torch/CUDA check FAILED — env is not usable"; exit 1 }
& $py -c "import bitsandbytes as bnb; print('bitsandbytes', bnb.__version__)"
if ($LASTEXITCODE -ne 0) { Write-Error "bitsandbytes check FAILED — env is not usable"; exit 1 }

Write-Host "`n✅ env ready. Next:"
Write-Host "   .\.venv\Scripts\python.exe src\prepare_data.py --dataset Salesforce/xlam-function-calling-60k"
Write-Host "   .\.venv\Scripts\python.exe src\train.py   --config config\qlora.yaml"
Write-Host "   .\.venv\Scripts\python.exe src\eval.py    --config config\qlora.yaml"
