cat << 'EOF' > run_inference.sh
#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate deepfake

# Thư mục chứa các file .pth đã train xong (ông sửa lại nếu lưu ở chỗ khác nhé)
CHECKPOINT_DIR="./checkpoints"

# Danh sách các dataset cần test
DATASETS=("faceforensic++" "celebdf" "dfdc" "wilddf")

# Danh sách các mô hình ablation và cờ tương ứng
declare -A MODELS
MODELS=(
    ["v2_baseline"]="GranIT_Ablation_BEST_AUC.pth" # Chú ý sửa tên file pth cho khớp với tên ông save
    ["v2_lhpf"]="GranIT_Ablation_BEST_AUC.pth"
    ["v2_fgafc"]="GranIT_Ablation_BEST_AUC.pth"
    ["v2_full"]="GranIT_Ablation_BEST_AUC.pth"
    ["v2_no_m"]="GranIT_Ablation_BEST_AUC.pth"
    ["only_global"]="GranIT_GlobalOnly_BEST_AUC.pth"
    ["only_local"]="GranIT_LocalOnly_BEST_AUC.pth"
    ["only_micro"]="GranIT_MicroOnly_BEST_AUC.pth"
    ["local_micro"]="GranIT_Local_Micro_BEST_AUC.pth"
)

echo "======================================================"
echo "BẮT ĐẦU QUÁ TRÌNH AUTO-INFERENCE"
echo "======================================================"

for model_flag in "${!MODELS[@]}"; do
    model_file="${MODELS[$model_flag]}"
    model_path="$CHECKPOINT_DIR/$model_file"

    if [ ! -f "$model_path" ]; then
        echo "[BỎ QUA] Không tìm thấy file weights cho $model_flag tại $model_path"
        continue
    fi

    echo "------------------------------------------------------"
    echo "Đang test mô hình: $model_flag"
    
    for dataset in "${DATASETS[@]}"; do
        echo " -> Trên dataset: $dataset"
        
        python ablation_inference.py \
            --ablation_model "$model_flag" \
            --dataset "$dataset" \
            --model_path "$model_path" \
            --batch_size 16 \
            --vis_dir "./visualizations/${model_flag}_${dataset}" \
            --log_file "results_${model_flag}.txt"
            
    done
done

echo "======================================================"
echo "TESTING HOÀN TẤT! HÃY KIỂM TRA CÁC FILE .CSV VÀ .TXT"
echo "======================================================"
