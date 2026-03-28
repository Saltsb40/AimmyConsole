set "VENV_PATH=%~dp0..\..\..\..\venv\Scripts\python.exe"
echo [Batch] Exporting Best Model to ONNX...

if exist "%VENV_PATH%" (
    "%VENV_PATH%" train_yolo.py --export-only %*
) else (
    python train_yolo.py --export-only %*
)

echo.
pause
