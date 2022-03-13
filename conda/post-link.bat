@echo off
setlocal enabledelayedexpansion

FOR %%F in ("%PREFIX%\Scripts\mcquic*-script.py") DO (
    call :modifyShebang %%F
)
echo.&pause&goto:eof

:modifyShebang
FOR /F "usebackq delims=" %%i IN (%~1) DO (
    if not defined shebang (
        set "shebang=1"
        echo.%PREFIX%\python -O>"%~1"
    ) else (
        echo.%%i>>"%~1"
    )
)
goto:eof
