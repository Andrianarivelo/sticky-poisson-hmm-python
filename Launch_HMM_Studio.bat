@echo off
REM ============================================================
REM  HMM Studio launcher.
REM
REM  Order of resolution for the Python interpreter:
REM    1. %HMM_PY%               (set this env var to override)
REM    2. The active conda env   (%CONDA_PREFIX%\python.exe)
REM    3. The first candidate below that exists.
REM
REM  Add or reorder candidates if you prefer a specific env.
REM ============================================================
setlocal

if defined HMM_PY (
    set "PY=%HMM_PY%"
    goto :run
)

if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "PY=%CONDA_PREFIX%\python.exe"
        goto :run
    )
)

for %%P in (
    "C:\ProgramData\anaconda3\python.exe"
    "%USERPROFILE%\.conda\envs\pyBer\python.exe"
    "%USERPROFILE%\AppData\Local\miniforge3\python.exe"
) do (
    if exist %%~P (
        set "PY=%%~P"
        goto :run
    )
)

echo Could not find a usable Python.
echo Set HMM_PY to your interpreter, e.g.:
echo     set HMM_PY=C:\path\to\python.exe
echo     Launch_HMM_Studio.bat
pause
exit /b 1

:run
echo Using Python: %PY%
"%PY%" "%~dp0hmm_studio\run_studio.py" %*
if errorlevel 1 (
    echo.
    echo HMM Studio exited with an error.
    pause
)
endlocal
