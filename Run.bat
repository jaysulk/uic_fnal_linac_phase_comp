@echo off
REM Set the path to your Anaconda installation
SET "CONDA_PATH=C:\Users\rithi\anaconda3"

REM Initialize Conda
CALL "%CONDA_PATH%\Scripts\activate.bat"

REM Check if environment exists
CALL conda info --envs | findstr dpienv >nul
IF ERRORLEVEL 1 (
    REM Create the environment
    CALL conda create -n dpienv python=3.10 -y
    CALL conda activate dpienv
    REM Install packages only once after creation
    pip install numpy pandas matplotlib scikit-learn torch imageio scipy nbconvert nbformat
) ELSE (
    REM Just activate if already exists
    CALL conda activate dpienv
)

REM Run your Python notebook automation script
python Train.py

:MENU
ECHO.
ECHO Press [E] to Exit
ECHO Press [1] to Re-Train
ECHO Press [2] to Test
set /p choice="> "

IF /i "%choice%"=="1" GOTO LOOP1
IF /i "%choice%"=="2" GOTO LOOP2
IF /i "%choice%"=="E" GOTO END
GOTO MENU

:LOOP1
python Train.py
GOTO MENU

:LOOP2
python Test.py
GOTO MENU

:END
ECHO Exiting...
pause