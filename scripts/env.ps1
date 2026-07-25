# Dot-source before any run:   . .\scripts\env.ps1
#
# Two things Python needs that Windows does NOT hand it automatically:
#   1. HF cache off C:\ — the default (%USERPROFILE%\.cache) fills a small system
#      drive fast; a 7B checkpoint is ~15 GB.
#   2. Proxy env vars — PowerShell honours the WinINET system proxy, but
#      requests/huggingface_hub only read HTTP_PROXY/HTTPS_PROXY, so downloads
#      die with SSLEOFError even while the browser works fine.

$env:HF_HOME = "E:\hf-cache"

# Pick up the system proxy (e.g. Clash on 127.0.0.1:7897) if one is configured.
$ie = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue
if ($ie.ProxyEnable -eq 1 -and $ie.ProxyServer) {
    $proxy = $ie.ProxyServer
    if ($proxy -notmatch '^https?://') { $proxy = "http://$proxy" }
    $env:HTTP_PROXY = $proxy
    $env:HTTPS_PROXY = $proxy
    Write-Host "proxy  -> $proxy"
}
# Mirror fallback for when huggingface.co is unreachable and there is no proxy:
#   $env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host "HF_HOME -> $env:HF_HOME"
