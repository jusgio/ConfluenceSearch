REM If the virtual environment exists, deactivate it and then delete it
if exist .venv (
    call .venv\Scripts\deactivate
    rmdir /s /q .venv
)

REM Create and activate the virtual environment
py -3.11 -m venv .venv
call .venv\Scripts\activate

REM Upgrade pip, as some modules (such as PyQt5) may not work with the current version
python -m pip install --upgrade pip

REM Install the requirements
pip install -r requirements.txt
