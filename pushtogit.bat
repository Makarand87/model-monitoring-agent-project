@echo off
git add .
set /p msg=Enter commit message: 
@REM set git commit -m "Auto-commit: %date% %time%"
git commit -m "%msg%"
