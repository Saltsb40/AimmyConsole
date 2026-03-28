@echo off
set "VENV_PATH=%~dp0..\..\..\..\venv\Scripts\python.exe"
echo [Batch] Starting YOLOv8 Model Training...

if exist "%VENV_PATH%" (
    echo [Batch] Using Virtual Environment...
    "%VENV_PATH%" train_yolo.py %*
) else (
    echo [Batch] WARNING: Virtual environment not found, falling back to system python.
    python train_yolo.py %*
)

echo.
pause
