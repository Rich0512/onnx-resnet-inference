@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   ResNet Inference Engine - Build Script
echo ========================================
echo.

:: Check for GPU flag
set USE_CUDA_FLAG=OFF
if /i "%1"=="--gpu" set USE_CUDA_FLAG=ON
if /i "%1"=="-g" set USE_CUDA_FLAG=ON

if "%USE_CUDA_FLAG%"=="ON" (
    echo   Build Mode: GPU ^(CUDA^)
) else (
    echo   Build Mode: CPU only
    echo   Use: build.bat --gpu for GPU build
)
echo.

:: Navigate to project root directory
cd /d "%~dp0.."

:: Default values
set "VS_VCVARS_PATH=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
set "CMAKE_GENERATOR=Visual Studio 17 2022"
set "CMAKE_ARCHITECTURE=x64"

:: Load .env file
if exist .env (
    echo [0/4] Loading configuration from .env...
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%b"=="" (
            set "%%a=%%b"
        )
    )
)

:: Convert forward slashes to backslashes for VS path
set "VS_VCVARS_PATH=!VS_VCVARS_PATH:/=\!"

:: Set Visual Studio environment
if exist "!VS_VCVARS_PATH!" (
    call "!VS_VCVARS_PATH!"
) else (
    echo ERROR: Cannot find vcvars64.bat at: !VS_VCVARS_PATH!
    pause
    exit /b 1
)

:: Navigate back to project directory
cd /d "%~dp0.."

:: Create build directory
if not exist build mkdir build
cd build

:: Clean previous build
echo [1/4] Cleaning previous build...
if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

:: Configure with CMake
echo [2/4] Configuring with CMake...
cmake -G "!CMAKE_GENERATOR!" -A !CMAKE_ARCHITECTURE! -DUSE_CUDA=!USE_CUDA_FLAG! ..

:: Build
echo [3/4] Building...
cmake --build . --config Release

:: Copy files
echo.
echo [4/4] Copying files...
if exist "..\models\resnet18.onnx" copy "..\models\resnet18.onnx" "Release\" 2>nul
if exist "..\data\test_cat.jpg" copy "..\data\test_cat.jpg" "Release\" 2>nul

echo.
echo ========================================
if "%USE_CUDA_FLAG%"=="ON" (
    echo   Build Complete! ^(GPU Mode^)
) else (
    echo   Build Complete! ^(CPU Mode^)
)
echo   Executable: %~dp0..\build\Release\main.exe
echo ========================================
echo.
pause

endlocal
