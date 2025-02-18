@echo off

REM create output folder
if not exist "..\output" (
    mkdir "..\output"
    echo "output folder created"
) else (
    echo "output folder already exists"
)

REM create results of decomposition folder
if not exist "..\output\results of decomposition" (
    mkdir "..\output\results of decomposition"
    echo "results of decomposition folder created"
) else (
    echo "results of decomposition folder already exists"
)

REM move Decomposition related Excel
move "*Decomposition*.xlsx" "..\output\results of decomposition\" >nul

REM create results of validation folder
if not exist "..\output\results of validation" (
    mkdir "..\output\results of validation"
    echo "results of validation folder created"
) else (
    echo "results of validation folder already exists"
)

REM move validation_result.xlsx file
move "validation_result*" "..\output\results of validation\" >nul

REM create results of prediction folder
if not exist "..\output\results of prediction" (
    mkdir "..\output\results of prediction"
    echo "results of prediction folder created"
) else (
    echo "results of prediction folder already exists"
)

REM move all other  .csv and .xlsx files to results of prediction folder
move "*.csv" "..\output\results of prediction\" >nul
move "*.xlsx" "..\output\results of prediction\" >nul

pause