@echo off
echo ========================================
echo   ResNet Inference Engine
echo ========================================
echo.

:: Navigate to project root, then Release folder
cd /d "%~dp0.."
cd build\Release

:: Check if executable exists
if not exist main.exe (
    echo ERROR: main.exe not found!
    echo Please run scripts\build.bat first.
    pause
    exit /b 1
)

:: Check if model exists
if not exist resnet18.onnx (
    echo Copying model file...
    copy "..\..\models\resnet18.onnx" "." 2>nul
)

:: Check if test image exists, download if not
if not exist test_cat.jpg (
    echo Downloading test image...
    powershell -Command "Invoke-WebRequest -Uri 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/600px-Cat_November_2010-1a.jpg' -OutFile 'test_cat.jpg'"
)

:: Run with default test image or provided image
if "%~1"=="" (
    echo.
    echo Running inference on test_cat.jpg...
    main.exe test_cat.jpg
) else (
    echo.
    echo Running inference on %~1...
    main.exe "%~1"
)

echo.
pause
