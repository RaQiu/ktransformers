@echo off
REM Install git hooks from kt-kernel\.githooks into the monorepo's .git\hooks
REM Windows equivalent of install-git-hooks.sh

setlocal enabledelayedexpansion

REM This script lives in kt-kernel\scripts\, so REPO_ROOT = kt-kernel
set "REPO_ROOT=%~dp0.."
set "HOOKS_SRC=%REPO_ROOT%\.githooks"

REM Detect the top-level Git worktree
for /f "delims=" %%G in ('git rev-parse --show-toplevel 2^>nul') do set "GIT_TOP=%%G"
if not defined GIT_TOP (
    echo [install-git-hooks] Not inside a git worktree; skipping. >&2
    exit /b 0
)

REM Normalize forward slashes from git output to backslashes
set "GIT_TOP=%GIT_TOP:/=\%"

if not exist "%GIT_TOP%\.git" (
    echo [install-git-hooks] Not inside a git worktree; skipping. >&2
    exit /b 0
)

set "HOOKS_DEST=%GIT_TOP%\.git\hooks"

if not exist "%HOOKS_SRC%" (
    echo [install-git-hooks] No .githooks directory found at %HOOKS_SRC% >&2
    exit /b 1
)

echo [install-git-hooks] Installing hooks from %HOOKS_SRC% to %HOOKS_DEST%

if not exist "%HOOKS_DEST%" mkdir "%HOOKS_DEST%"

for %%F in ("%HOOKS_SRC%\*") do (
    copy /y "%%F" "%HOOKS_DEST%\%%~nxF" >nul
    echo   copied %%~nxF
)

echo [install-git-hooks] Done. Hooks installed.
endlocal
