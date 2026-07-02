

@echo off
echo Creating Python environment and installing requirements...
py -m pip install -r requirements.txt

echo.
echo Preparing CDC BRFSS heart attack dataset...
py scripts\prepare_dataset.py

echo.
echo Training baseline models...
py scripts\train_baseline_models.py

echo.
echo Done. Check data\processed and outputs folders.
pause

