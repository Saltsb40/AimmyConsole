@echo off
set "VENV_PATH=%~dp0..\..\..\..\venv\Scripts\python.exe"
echo [Batch] Exporting Best Model to ONNX...

if exist "%VENV_PATH%" (
    echo [Batch] Using Virtual Environment...
    setlocal enabledelayedexpansion
    
    :: Check if --imgsz is already in arguments
    set "HAS_IMSZ=0"
    for %%a in (%*) do if /i "%%a"=="--imgsz" set "HAS_IMSZ=1"
    
    if !HAS_IMSZ! equ 0 (
        echo.
        echo [Batch] Select Export Resolution:
        echo  [1] 320px (Fastest^)
        echo  [2] 640px (Standard^)
        echo  [3] 1024px (High^)
        echo  [4] 1280px (Max^)
        set /p IMSZ_CHOICE=" [Input] Choice [1-4, Default 2]: "
        
        set "IMSZ=640"
        if "!IMSZ_CHOICE!"=="1" set "IMSZ=320"
        if "!IMSZ_CHOICE!"=="2" set "IMSZ=640"
        if "!IMSZ_CHOICE!"=="3" set "IMSZ=1024"
        if "!IMSZ_CHOICE!"=="4" set "IMSZ=1280"
        
        "%VENV_PATH%" train_yolo.py --export-only --imgsz !IMSZ! %*
    ) else (
        "%VENV_PATH%" train_yolo.py --export-only %*
    )
    endlocal
) else (
    echo [Batch] WARNING: Virtual environment not found, falling back to system python.
    python train_yolo.py --export-only %*
)

echo.
pause
