"""
圖像標註服務範例使用腳本
"""
from app.services import ImageAnnotationService
from app.core.logger import get_logger

logger = get_logger(__name__)


def main():
    """主函式"""
    try:
        # 初始化圖像標註服務
        logger.info("正在初始化圖像標註服務...")
        service = ImageAnnotationService()
        
        # 設定要處理的圖像
        image_path = "data/uploads/截圖 2025-11-14 14.49.54.png"
        target_item = "play audio button"
        
        # 方法 1: 直接標註並儲存
        logger.info("=== 方法 1: 標註並儲存圖像 ===")
        saved_files = service.annotate_image(
            image_path=image_path,
            target_item=target_item
        )
        
        logger.info(f"已儲存 {len(saved_files)} 個標註圖像：")
        for file_path in saved_files:
            logger.info(f"  - {file_path}")
        
        # 方法 2: 僅獲取偵測摘要（不儲存圖像）
        logger.info("\n=== 方法 2: 獲取偵測摘要 ===")
        summary = service.get_detection_summary(
            image_path=image_path,
            target_item=target_item
        )
        
        logger.info(f"偵測摘要：")
        logger.info(f"  圖像路徑: {summary['image_path']}")
        logger.info(f"  目標物件: {summary['target_item']}")
        logger.info(f"  偵測數量: {summary['total_detected']}")
        logger.info(f"  偵測物件:")
        for i, obj in enumerate(summary['objects'], 1):
            logger.info(f"    {i}. {obj['label']}")
            logger.info(f"       座標: {obj['box_2d']}")
        
        # 方法 3: 只偵測物件（不繪製和儲存）
        logger.info("\n=== 方法 3: 僅偵測物件 ===")
        detected_objects = service.detect_objects(
            image_path=image_path,
            target_item=target_item
        )
        
        logger.info(f"偵測到 {len(detected_objects)} 個物件：")
        for i, obj in enumerate(detected_objects, 1):
            logger.info(f"  {i}. {obj.label} - 座標: {obj.box_2d}")
        
    except Exception as e:
        logger.error(f"執行失敗：{e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
