@echo off
REM Create virtual environment
echo Creating virtual environment...
py -3.12 -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install required packages
pip install sentence-transformers faiss-cpu pyqt5 requests beautifulsoup4

echo.
echo Setup complete. To activate the environment later, run:
echo   .venv\Scripts\activate
echo.
pause
