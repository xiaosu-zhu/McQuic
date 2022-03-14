@echo off
setlocal enabledelayedexpansion

FOR %%F in ("C:\Users\Xiaos\anaconda3\envs\mcquic\Scripts\mcquic*-script.py") DO (
    echo %%F
    call :modifyShebang %%F
)
echo.&pause&goto:eof

:modifyShebang
echo.#^^!%PREFIX%\python.exe -O > "%~1.new"
type "%~1" >> "%~1.new"
type "%~1.new" > "%~1"
del "%~1.new"
goto:eof
