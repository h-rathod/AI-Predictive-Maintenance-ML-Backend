@echo off
echo Creating resources directories...
mkdir src\main\resources\model 2>nul
mkdir src\main\resources\data 2>nul

echo Copying model files...
copy model\*.* src\main\resources\model\

echo Copying data files...
copy data\*.* src\main\resources\data\

echo Resources copied successfully! 