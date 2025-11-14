"""
測試圖片標註功能
"""
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.annotating_service import ImageAnnotationService
from app.config import get_settings

def test_annotation_service():
    """測試圖片標註服務"""
    print("=" * 50)
    print("圖片標註服務測試")
    print("=" * 50)
    
    settings = get_settings()
    
    # 檢查 Google API Key
    if not settings.google_api_key:
        print("❌ 錯誤：未設定 GOOGLE_API_KEY")
        print("請在 .env 檔案中設定 GOOGLE_API_KEY")
        return False
    
    print(f"✅ Google API Key 已設定")
    
    # 初始化服務
    try:
        service = ImageAnnotationService()
        print(f"✅ 圖片標註服務初始化成功")
        print(f"   使用模型：{service.model}")
        return True
    except Exception as e:
        print(f"❌ 服務初始化失敗：{e}")
        return False


def test_detect_objects(image_path: str, target_item: str = "objects"):
    """測試物件偵測"""
    print("\n" + "=" * 50)
    print("物件偵測測試")
    print("=" * 50)
    
    if not Path(image_path).exists():
        print(f"❌ 圖片不存在：{image_path}")
        return
    
    try:
        service = ImageAnnotationService()
        
        print(f"📷 圖片路徑：{image_path}")
        print(f"🎯 偵測目標：{target_item}")
        print("⏳ 正在偵測物件...")
        
        detected_objects = service.detect_objects(image_path, target_item)
        
        print(f"\n✅ 偵測完成！共找到 {len(detected_objects)} 個物件：")
        for i, obj in enumerate(detected_objects, 1):
            print(f"   {i}. {obj.label}")
            print(f"      座標：{obj.box_2d}")
        
        # 標註圖片
        print("\n⏳ 正在標註圖片...")
        annotated_files = service.annotate_image(image_path, target_item)
        
        print(f"\n✅ 標註完成！已儲存 {len(annotated_files)} 個檔案：")
        for file in annotated_files:
            print(f"   📁 {file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 偵測失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 開始測試圖片標註功能\n")
    
    # 測試服務初始化
    if not test_annotation_service():
        print("\n❌ 服務初始化失敗，停止測試")
        sys.exit(1)
    
    # 如果有提供圖片路徑，進行物件偵測測試
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        target_item = sys.argv[2] if len(sys.argv) > 2 else "objects"
        test_detect_objects(image_path, target_item)
    else:
        print("\n💡 提示：")
        print("   若要測試物件偵測，請提供圖片路徑：")
        print("   python test_image_annotation.py <圖片路徑> [偵測目標]")
        print("\n   範例：")
        print("   python test_image_annotation.py ./test_image.jpg person")
    
    print("\n" + "=" * 50)
    print("測試完成！")
    print("=" * 50)
