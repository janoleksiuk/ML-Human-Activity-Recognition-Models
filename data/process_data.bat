@echo off
REM Batch script to run data_preprocessing.py for different activities

REM Ask the user for the activity name
set /p activity="Enter activity name (e.g. drinking, eating, running): "

REM Run the Python script with the given activity
python data_preprocessing.py ^
    --input_dir raw/%activity% ^
    --output_path processed/%activity%.npz ^
    --class_name %activity% ^
    --smooth sg ^
    --seq_len 30

pause