@echo OFF

REM === Cesta k instalaci Anacondy ===
set "CONDAPATH=C:\ProgramData\anaconda3"

REM === Název a cesta k prostředí ===
set "ENVNAME=virtual_sketching"
set "ENVPATH=%USERPROFILE%\.conda\envs\%ENVNAME%"

REM === Aktivace prostředí ===
call "%CONDAPATH%\Scripts\activate.bat" "%ENVPATH%"

REM === Spuštění GUI ===
python virtual_sketch_gui.py

REM === Pozastavení po ukončení ===
echo.
pause

REM === Deaktivace prostředí ===
call conda deactivate
