@echo off
REM Spider Guardian Advocacy Automation Script
REM Run this daily via Windows Task Scheduler

echo ============================================================
echo   SPIDER GUARDIAN - Advocacy Power-Up Cycle
echo ============================================================
echo.

cd /d "%~dp0"

REM Activate Python environment if you have one
REM call venv\Scripts\activate

echo [1/4] Refreshing reply metrics...
python -m spider_guardian.scripts.refresh_my_replies
if errorlevel 1 (
    echo ERROR: Failed to refresh reply metrics
    pause
    exit /b 1
)

echo.
echo [2/4] Updating LangSmith datasets...
python -m spider_guardian.langsmith.config update_dataset_from_db --dataset-name trending-dataset --max-examples 500
if errorlevel 1 (
    echo ERROR: Failed to update datasets
    pause
    exit /b 1
)

echo.
echo [3/4] Running engagement analysis...
python -m spider_guardian.scripts.advocacy_orchestrator --analyze-engagement
if errorlevel 1 (
    echo ERROR: Failed to analyze engagement
    pause
    exit /b 1
)

echo.
echo [4/4] Generating trend reports...
python -m spider_guardian.scripts.analyze_trends
if errorlevel 1 (
    echo ERROR: Failed to generate trends
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   SUCCESS! All advocacy power-ups completed
echo ============================================================
echo.
echo Check your results:
echo   - Metrics: spider_guardian.sqlite
echo   - Reports: figures\engagement_analysis\
echo   - Trends: figures\advocacy_trends\
echo.
echo Press any key to exit...
pause >nul
