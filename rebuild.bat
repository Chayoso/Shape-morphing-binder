@echo off
chcp 65001 > nul
cd /d C:\dev\shape-morphing_v2.3.2
call C:\Users\ok429\anaconda3\envs\diffmpm_v2.1.0\Scripts\activate.bat
python -m pip install -e . --no-build-isolation --force-reinstall
