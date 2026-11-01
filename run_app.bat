@echo off
title MovieRec AI - Recommendation Engine (HCLTech Project #28)
echo ======================================================================
echo   Launching MovieRec AI - Collaborative Filtering Engine
echo   HCLTech Career Shaper Programme (Project #28)
echo ======================================================================
echo.

cd /d "C:\Users\HP\OneDrive\Desktop\HCL Project"

echo [1/2] Starting Flask Web Server on http://127.0.0.1:5000 ...
echo [2/2] Opening web browser in 3 seconds...

start "" http://127.0.0.1:5000

"C:\Users\HP\AppData\Local\Microsoft\WindowsApps\python.exe" app.py

pause
