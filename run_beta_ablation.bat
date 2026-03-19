@echo off
setlocal enabledelayedexpansion

@REM echo TRAINING CROSS-SCALE MODEL
@REM ::train xong 0.8 -> 1.3 r, con lai 1.4 vs 1.5
@REM set MARGINS_TRAINING=1.4 1.5 
@REM set EXP_PREFIX=vit_beta

@REM for %%b in (%MARGINS_TRAINING%) do (
@REM     echo  TRAINING MARGIN = %%b 

@REM     python ablation_training.py --ablation_model margin --crop_margin %%b --save_name  %EXP_PREFIX%_%%b --batch_size 32

@REM     echo  -^> DONE MARGIN: %%b
@REM     echo.
@REM )

@REM echo DONE 6 phases trainign


echo ========================================================
echo   INFERENCE CROSS-MARGIN EVALUATION (8x8 MATRIX)
echo ========================================================

set MARGINS=0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5

for %%m in (%MARGINS%) do (
    echo.
    echo ========================================================
    echo  EVALUATING MODEL TRAIN ON MARGIN: %%m
    echo ========================================================
    
    for %%b in (%MARGINS%) do (
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