@REM @REM python train.py --model_type baseline --save_name tight_test --epochs 20 --data_dir ./data/faces_tight_split
@REM @REM python train.py --model_type baseline --save_name wide_test --epochs 20 --data_dir ./data/faces_wide_split


@REM @REM python eval.py --model_path ./models/tight_test.pth --model_type baseline --test_dir ./data/faces_tight_split/test
@REM @REM python eval.py --model_path ./models/tight_test.pth --model_type baseline --test_dir ./data/faces_wide_split/test
@REM @REM python eval.py --model_path ./models/wide_test.pth --model_type baseline --test_dir ./data/faces_tight_split/test
@REM @REM python eval.py --model_path ./models/wide_test.pth --model_type baseline --test_dir ./data/faces_wide_split/test


@REM @REM python eval.py --model_path ./models/tight_test.pth --model_type baseline --test_dir D:/Project/model_testing/data/test/frames/DFDC
@REM @REM python eval.py --model_path ./models/wide_test.pth --model_type baseline --test_dir D:/Project/model_testing/data/test/frames/DFDC

@REM @REM python eval.py --model_path ./models/tight_test.pth --model_type baseline --test_dir D:/Project/model_testing/data/test/frames/CelebDF
@REM @REM python eval.py --model_path ./models/wide_test.pth --model_type baseline --test_dir D:/Project/model_testing/data/test/frames/CelebDF

@REM @REM python train_baseline.py --model_type tight --save_name exp1
@REM @REM python train_baseline.py --model_type wide --save_name exp1

@REM @REM python evaluate_baseline.py --model_type tight --model_path exp1_tight_epoch_10.pth
@REM @REM python evaluate_baseline.py --model_type wide --model_path exp1_wide_epoch_10.pth

@REM @REM python cross_evaluate.py --model_type tight --model_path models/exp1_tight_epoch_10.pth --data_dir ./data/celebdf_processed
@REM @REM python cross_evaluate.py --model_type wide --model_path models/exp1_wide_epoch_10.pth --data_dir ./data/celebdf_processed
@REM @REM python cross_evaluate.py --model_type afc --model_path models/afc_model_epoch_10.pth --data_dir ./data/celebdf_processed

@REM @REM python train_baseline.py --model_type tight --batch_size 16 --data_dir ./data/faces_tight
@REM @REM python train_baseline.py --model_type wide --batch_size 16 --data_dir ./data/faces_wide

@REM @REM python train_sota.py --model_type efficientnet --batch_size 16
@REM @REM python train_sota.py --model_type resnet50 --batch_size 16

@REM python cross_evaluate.py --model_type tight --model_path models/baseline_tight_vit_lora_BEST.pth --data_dir data/test/wilddf_processed
@REM python cross_evaluate.py --model_type wide --model_path models/baseline_wide_vit_lora_BEST.pth --data_dir data/test/wilddf_processed
@REM python cross_evaluate.py --model_type afc --model_path models/fg_afc_vit_lora_BEST.pth --data_dir data/test/wilddf_processed

@REM python test_sota.py --model_type xception --model_path models/sota_xception_BEST.pth --data_dir data/test/wilddf_processed
@REM python test_sota.py --model_type efficientnet --model_path models/sota_efficientnet_BEST.pth --data_dir data/test/wilddf_processed
@REM python test_sota.py --model_type resnet50 --model_path models/sota_resnet50_BEST.pth --data_dir data/test/wilddf_processed



:: GranIT

@REM python train.py

@REM python inference.py --dataset dfdc           --model_path checkpoints/GranIT_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_AUC/FaceForensic++
@REM python evaluate_baseline.py
@REM python inference.py --dataset faceforensic++ --model_path checkpoints/GranIT_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_AUC/FaceForensic++
@REM python inference.py --dataset celebdf        --model_path checkpoints/GranIT_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_AUC/CelebDF
@REM python inference.py --dataset wilddf         --model_path checkpoints/GranIT_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_AUC/WildDF

@REM python inference.py --dataset faceforensic++ --model_path checkpoints/GranIT_BEST_VAL_LOSS.pth --batch_size 8 --vis_dir ./visualizations/GranIT_VAL_LOSS/FaceForensic++
@REM python inference.py --dataset celebdf        --model_path checkpoints/GranIT_BEST_VAL_LOSS.pth --batch_size 8 --vis_dir ./visualizations/GranIT_VAL_LOSS/CelebDF
@REM python inference.py --dataset wilddf         --model_path checkpoints/GranIT_BEST_VAL_LOSS.pth --batch_size 8 --vis_dir ./visualizations/GranIT_VAL_LOSS/WildDF

:: ABALTION STUDYGranIT_GlobalOnly
@REM python ablation_training.py --ablation_model only_global --save_name only_global_model
@REM python ablation_training.py --ablation_model only_local --save_name only_local_model
@REM python ablation_training.py --ablation_model only_micro --save_name only_micro_model
@REM python ablation_training.py --ablation_model local_micro --save_name local_micro_model
@REM python ablation_training.py --ablation_model global_local --save_name global_local_model

:: ONLY GLOBAL
@REM python ablation_inference.py --ablation_model only_global --dataset faceforensic++ --model_path checkpoints/GranIT_GlobalOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_GlobalOnly_BEST_AUC/FaceForensic++
@REM python ablation_inference.py --ablation_model only_global --dataset celebdf        --model_path checkpoints/GranIT_GlobalOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_GlobalOnly_BEST_AUC/CelebDF
@REM python ablation_inference.py --ablation_model only_global --dataset wilddf         --model_path checkpoints/GranIT_GlobalOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_GlobalOnly_BEST_AUC/WildDF

@REM python ablation_inference.py --ablation_model only_global --dataset dfdc         --model_path checkpoints/GranIT_GlobalOnly_BEST_AUC.pth --batch_size 32 --vis_dir ./visualizations/GranIT_GlobalOnly_BEST_AUC/WildDF
::ONLY LOCAL
@REM python ablation_inference.py --ablation_model only_local --dataset faceforensic++ --model_path checkpoints/GranIT_LocalOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_LocalOnly_BEST_AUC/FaceForensic++
@REM python ablation_inference.py --ablation_model only_local --dataset celebdf        --model_path checkpoints/GranIT_LocalOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_LocalOnly_BEST_AUC/CelebDF
@REM python ablation_inference.py --ablation_model only_local --dataset wilddf         --model_path checkpoints/GranIT_LocalOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_LocalOnly_BEST_AUC/WildDF

@REM python ablation_inference.py --ablation_model only_local --dataset dfdc         --model_path checkpoints/GranIT_LocalOnly_BEST_AUC.pth --batch_size 32 --vis_dir ./visualizations/GranIT_LocalOnly_BEST_AUC/WildDF
::ONLY MICRO
@REM python ablation_inference.py --ablation_model only_micro --dataset faceforensic++ --model_path checkpoints/GranIT_MicroOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_MicroOnly_BEST_AUC/FaceForensic++
@REM python ablation_inference.py --ablation_model only_micro --dataset celebdf        --model_path checkpoints/GranIT_MicroOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_MicroOnly_BEST_AUC/CelebDF
@REM python ablation_inference.py --ablation_model only_micro --dataset wilddf         --model_path checkpoints/GranIT_MicroOnly_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_MicroOnly_BEST_AUC/WildDF

@REM python ablation_inference.py --ablation_model only_micro --dataset dfdc         --model_path checkpoints/GranIT_MicroOnly_BEST_AUC.pth --batch_size 32 --vis_dir ./visualizations/GranIT_MicroOnly_BEST_AUC/WildDF
::LOCAL + MICRO
@REM python ablation_inference.py --ablation_model local_micro --dataset faceforensic++ --model_path checkpoints/GranIT_Local_Micro_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Local_Micro_BEST_AUC/FaceForensic++
@REM python ablation_inference.py --ablation_model local_micro --dataset celebdf        --model_path checkpoints/GranIT_Local_Micro_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Local_Micro_BEST_AUC/CelebDF
@REM python ablation_inference.py --ablation_model local_micro --dataset wilddf         --model_path checkpoints/GranIT_Local_Micro_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Local_Micro_BEST_AUC/WildDF
@REM python ablation_inference.py --ablation_model local_micro --dataset dfdc         --model_path checkpoints/GranIT_Local_Micro_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Local_Micro_BEST_AUC/WildDF
::GLOBAL + LOCAL
@REM python ablation_inference.py --ablation_model global_local --dataset faceforensic++ --model_path checkpoints/GranIT_Global_Local_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Global_Local_BEST_AUC/FaceForensic++
@REM python ablation_inference.py --ablation_model global_local --dataset celebdf        --model_path checkpoints/GranIT_Global_Local_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Global_Local_BEST_AUC/CelebDF
@REM python ablation_inference.py --ablation_model global_local --dataset wilddf         --model_path checkpoints/GranIT_Global_Local_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Global_Local_BEST_AUC/WildDF
@REM python ablation_inference.py --ablation_model global_local --dataset dfdc         --model_path checkpoints/GranIT_Global_Local_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_Global_Local_BEST_AUC/WildDF
