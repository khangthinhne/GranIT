import os
import random
import shutil
from pathlib import Path

def split_ffpp_dataset(source_dir, output_dir, test_ratio=0.2, seed=42):
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 5 thư mục con trong FF++
    categories = ['original', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    # 1. Quét thư mục 'original' để lấy danh sách các Video ID (người gốc)
    original_dir = source_path / 'original'
    all_original_files = os.listdir(original_dir)
    
    # Lấy ID trước dấu '_' (vd: "001_000.jpg" -> "001")
    unique_ids = list(set([f.split('_')[0] for f in all_original_files if f.endswith('.jpg') or f.endswith('.png')]))
    unique_ids.sort()
    
    # 2. Xáo trộn và chia Split 80/20 dựa trên ID
    random.shuffle(unique_ids)
    test_count = int(len(unique_ids) * test_ratio)
    
    test_ids = set(unique_ids[:test_count])
    train_ids = set(unique_ids[test_count:])
    
    print(f"Tổng số IDs: {len(unique_ids)}")
    print(f"-> Train IDs: {len(train_ids)}")
    print(f"-> Test IDs: {len(test_ids)}")
    print("-" * 30)

    # 3. Tạo cấu trúc thư mục đích
    for split in ['train', 'test']:
        for cat in categories:
            os.makedirs(output_path / split / cat, exist_ok=True)

    # 4. Copy file theo ID đã chia
    for cat in categories:
        cat_dir = source_path / cat
        if not cat_dir.exists():
            continue
            
        print(f"Đang xử lý thư mục: {cat}...")
        files = os.listdir(cat_dir)
        
        for f in files:
            # Lấy ID của file hiện tại
            vid_id = f.split('_')[0]
            
            # Quyết định xem file này thuộc train hay test
            if vid_id in test_ids:
                dest = output_path / 'test' / cat / f
            elif vid_id in train_ids:
                dest = output_path / 'train' / cat / f
            else:
                continue # Bỏ qua nếu có file rác không xác định được ID
                
            # Dùng shutil.copy2 để copy (an toàn, không mất data cũ)
            # Nếu ổ cứng ông sắp đầy, có thể đổi thành shutil.move(cat_dir / f, dest)
            shutil.copy2(cat_dir / f, dest)
            
    print("HOÀN TẤT CHIA DỮ LIỆU!")

if __name__ == '__main__':
    # Đổi lại đường dẫn cho chuẩn với máy ông nha
    SOURCE = "data/faces_processed"
    OUTPUT = "data/faces_processed_split"
    
    split_ffpp_dataset(SOURCE, OUTPUT, test_ratio=0.2)