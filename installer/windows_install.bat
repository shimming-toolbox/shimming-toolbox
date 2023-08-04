@echo off
REM This script installs Shimming Toolbox on Windows. The script needs to be located withing the source files "shimming-toolbox\installer\"
setlocal enableextensions enabledelayedexpansion

REM The installation path cannot have spaces
REM Todo: Make the installation path a script argument so that users with spaces in their paths have an alternative

set "ST_DIR=%userprofile%\shimming-toolbox"
set "PYTHON_DIR=python"
for %%d in ("%~dp0..") do set "ST_SOURCE_FILES=%%~fd"
echo %ST_SOURCE_FILES%
pushd "%CD%"

echo Creating Shimming Toolbox directory
if not exist "%ST_DIR%" (mkdir "%ST_DIR%")
if exist "%ST_DIR%\%PYTHON_DIR%" (rmdir /s /q "%ST_DIR%\%PYTHON_DIR%" || goto error)
mkdir "%ST_DIR%\%PYTHON_DIR%"

echo Installing conda in %ST_DIR%\%PYTHON_DIR%
REM Test for other OSs
set "CONDA_INSTALLER=Mambaforge-Windows-x86_64.exe"
set "CONDA_INSTALLER_URL=https://github.com/conda-forge/miniforge/releases/latest/download/%CONDA_INSTALLER%"

:uniqLoop
set "UNIQUE_TMP_INSTALLER=%tmp%\st_%RANDOM%_%CONDA_INSTALLER%"
if exist %UNIQUE_TMP_INSTALLER% (goto :uniqLoop)

REM Download mamba
powershell -Command "Invoke-WebRequest '%CONDA_INSTALLER_URL%' -OutFile '%UNIQUE_TMP_INSTALLER%'"

REM Install mamba
echo Installing ...
start /wait "" "%UNIQUE_TMP_INSTALLER%" /RegisterPython=0 /S /D=%ST_DIR%\%PYTHON_DIR%

del "%UNIQUE_TMP_INSTALLER%"

REM Installing dcm2niix and python
echo Installing dcm2niix and python
call "%ST_DIR%\%PYTHON_DIR%\condabin\mamba.bat" install -y -c conda-forge dcm2niix python=3.9 || goto error

REM Installing Shimming Toolbox
copy "%ST_SOURCE_FILES%\config\dcm2bids.json" "%ST_DIR%\dcm2bids.json" || goto error

cd "%ST_SOURCE_FILES%"
"%ST_DIR%\%PYTHON_DIR%\python.exe" -m pip install -e ".[docs,dev]" --no-warn-script-location || goto error

REM Create launchers for Shimming Toolbox
set "BIN_DIR=bin"
if exist "%ST_DIR%\%BIN_DIR%" (rmdir /s /q "%ST_DIR%\%BIN_DIR%" || goto error)
mkdir "%ST_DIR%\%BIN_DIR%"
for %%f in ("%ST_DIR%\%PYTHON_DIR%\Scripts\st_*.*") do (
	copy "%%f" "%ST_DIR%\%BIN_DIR%" || goto error
)
REM Add dcm2bids in the launchers (not currently accessible if not in the path)
for %%f in ("%ST_DIR%\%PYTHON_DIR%\Scripts\dcm2bids*") do (
	copy "%%f" "%ST_DIR%\%BIN_DIR%" || goto error
)

REM Add scripts to the User's path
for /F "skip=2 tokens=2,*" %%A in ('reg.exe query "HKEY_CURRENT_USER\Environment" /v path') do set "OLD_PATH=%%B"
REM If OLD_PATH is not an empty string
REM Try to remove the ST path from OLD_PATH
if not "%OLD_PATH%"=="" (call set "PATH_NO_ST=%%OLD_PATH:%ST_DIR%\%BIN_DIR%=%%") else (goto error)
REM IF OLD_PATH and PATH_NO_ST are the same, it means ST is not in the path
if "%PATH_NO_ST%"=="%OLD_PATH%" (
	echo "Adding ST bins to the path"
	if not "%OLD_PATH:~-1%"==";" (set "LIST_PATH=%OLD_PATH%;") else (set "LIST_PATH=%OLD_PATH%")
	set "NEW_ST_PATH=!LIST_PATH!%ST_DIR%\%BIN_DIR%\;"
	REG ADD "HKEY_CURRENT_USER\Environment" /v path /d "!NEW_ST_PATH!" /t REG_EXPAND_SZ /f
	)

echo To use Shimming Toolbox's scripts, either reboot your computer or follow these instructions:
echo:
echo Open the Start Menu -^> Type 'environment' -^> Open 'Edit environment variables for your account'
echo Click 'OK'
echo:
echo You can now access Shimming Toolbox from the command prompt

endlocal

echo "Installation completed successfully"
goto exit

:error
set "CACHED_ERROR_LEVEL=%errorlevel%"
echo Failed with error #%CACHED_ERROR_LEVEL%

:exit
echo "Exiting"
if "%CACHED_ERROR_LEVEL%"=="" (set "CACHED_ERROR_LEVEL=0")
popd
exit /b %CACHED_ERROR_LEVEL%
