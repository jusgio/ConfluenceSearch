REM arg 1: nom du script
SET filename=%1
SET filenameStem=%~n1

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

REM Installe pyinstaller
pip install pyinstaller

REM On supprime les folders dist et build, au besoin
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist %filenameStem%.spec del %filenameStem%.spec

REM On produit l'exec
pyinstaller --onefile --windowed %filename%


REM TODO: faire le ménage une fois le exe produit
REM .venv (deact avant), build, dist, %filenameStem%.spec
if exist .venv (
    call .venv\Scripts\deactivate
    rmdir /s /q .venv
)

REM on déplace le exe dans le répertoire courant
if exist dist\%filenameStem%.exe move dist\%filenameStem%.exe %filenameStem%.exe

if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist %filenameStem%.spec del %filenameStem%.spec

REM faire une fonction de delete