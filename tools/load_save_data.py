import os
import json

def load_json_data(file_path: str) -> list:
    """從指定的路徑載入 JSON 檔案。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{file_path}'。")
        return []
    except json.JSONDecodeError:
        print(f"錯誤：檔案 '{file_path}' 的 JSON 格式不正確。")
        return []

def save_to_json(data: list, output_file_path: str):
    """將處理好的資料儲存為 JSON 檔案。"""
    if not data:
        print("沒有資料可以儲存。")
        return
        
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"成功將 Embeddings 儲存至 '{output_file_path}'")
    except IOError as e:
        print(f"寫入檔案時發生錯誤：{e}")