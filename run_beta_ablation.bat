@echo off
setlocal enabledelayedexpansion


echo TRAINING CROSS-SCALE MODEL
set MARGINS=1.5
set MARGINS_testmodel= 1.4 1.5
set MARGINS_test=0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
set EXP_PREFIX=GranIT_Margin


for %%b in (%MARGINS%) do (
    echo  TRAINING MARGIN = %%b


    python ablation_training.py --ablation_model margin --crop_margin %%b --save_name  %EXP_PREFIX%_%%b --batch_size 32


    echo  -^> DONE MARGIN: %%b
    echo.
)


echo DONE 6 phases trainign




echo ========================================================
echo   INFERENCE CROSS-MARGIN EVALUATION (8x8 MATRIX)
echo ========================================================


for %%m in (%MARGINS_testmodel%) do (
    echo.
    echo ========================================================
    echo  EVALUATING MODEL TRAIN ON MARGIN: %%m
    echo ========================================================
   
    for %%b in (%MARGINS_test%) do (
        echo    -[TEST] Inference on Test Beta: %%b ...
       
        python ablation_inference.py ^
            --ablation_model margin ^
            --dataset faceforensic++ ^
            --crop_margin %%b ^
            --model_path checkpoints/GranIT_Margin_%%m_BEST_AUC.pth ^
            --batch_size 32 ^
            --vis_dir ./visualizations/GranIT_Margin_%%m/test_beta_%%b/
       
        echo    -^> XONG TEST BETA: %%b
    )
    echo =^> DONE EVALUATION ON MODEL MARGIN: %%m
)


echo.
echo DONE 8x8 PHASES


pause
