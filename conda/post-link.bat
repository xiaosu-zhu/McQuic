@echo off
setlocal enabledelayedexpansion

FOR %%F in ("%PREFIX%\bin\mcquic*-script.py") DO (
    call :modifyShebang %%F
)
echo.&pause&goto:eof

:modifyShebang
FOR /F "usebackq delims=" %%i IN (%~1) DO (
    if not defined shebang (
        set "shebang=1"
        echo.%%i -OO>"%~1"
    ) else (
        echo.%%i>>"%~1"
    )
)
goto:eof
