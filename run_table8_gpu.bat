@echo off
cd /d "%~dp0"
echo Table 8 子期间训练 (GPU) - Murray et al. 2024 Section 5.1
echo 设备: CUDA (RTX 3060)
echo.
python crsp_table8_stability.py ^
  --run-dir result/run_20260523_232834_crsp_paper_crsp_all_cnn_lstm_excess ^
  --n-ensemble 5 ^
  --device cuda
pause
