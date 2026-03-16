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

@REM python inference.py --dataset faceforensic++ --model_path checkpoints/GranIT_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_AUC/FaceForensic++
@REM python inference.py --dataset celebdf        --model_path checkpoints/GranIT_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_AUC/CelebDF
python inference.py --dataset wilddf         --model_path checkpoints/GranIT_BEST_AUC.pth --batch_size 8 --vis_dir ./visualizations/GranIT_AUC/WildDF

@REM python inference.py --dataset faceforensic++ --model_path checkpoints/GranIT_BEST_VAL_LOSS.pth --batch_size 8 --vis_dir ./visualizations/GranIT_VAL_LOSS/FaceForensic++
@REM python inference.py --dataset celebdf        --model_path checkpoints/GranIT_BEST_VAL_LOSS.pth --batch_size 8 --vis_dir ./visualizations/GranIT_VAL_LOSS/CelebDF
@REM python inference.py --dataset wilddf         --model_path checkpoints/GranIT_BEST_VAL_LOSS.pth --batch_size 8 --vis_dir ./visualizations/GranIT_VAL_LOSS/WildDF
