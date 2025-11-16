"""
圖像標註服務
使用 Google Gemini API 進行物件偵測和邊界框標註
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

from app.core.logger import get_logger
from app.core.exceptions import APIKeyError, ValidationError
from app.config import get_settings

logger = get_logger(__name__)


class DetectedObject:
    """偵測到的物件資料類別"""

    def __init__(self, box_2d: List[int], label: str):
        """
        初始化偵測物件

        Args:
            box_2d: 2D 邊界框座標 [y_min, x_min, y_max, x_max]，範圍 0-1000
            label: 物件標籤
        """
        self.box_2d = box_2d
        self.label = label

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {"box_2d": self.box_2d, "label": self.label}


class ImageAnnotationService:
    """圖像標註服務"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.0-flash-exp",
        max_image_size: Tuple[int, int] = (1024, 1024),
    ):
        """
        初始化圖像標註服務

        Args:
            api_key: Google API Key
            model: Gemini 模型名稱
            max_image_size: 圖像最大尺寸 (寬, 高)
        """
        settings = get_settings()
        self.api_key = api_key or settings.google_api_key
        self.model = model
        self.max_image_size = max_image_size

        if not self.api_key:
            raise APIKeyError("Google API Key 未設定")

        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"圖像標註服務已初始化，使用模型：{self.model}")

    def _parse_json_response(self, json_output: str) -> str:
        """
        解析 JSON 回應，移除 markdown 標記

        Args:
            json_output: 原始 JSON 輸出

        Returns:
            清理後的 JSON 字串
        """
        # 移除前後的空白
        json_output = json_output.strip()

        # 移除 markdown 代碼塊標記
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                json_output = "\n".join(lines[i + 1 :])
                output = json_output.split("```")[0].strip()
                return output
            elif line.strip() == "```":
                json_output = "\n".join(lines[i + 1 :])
                output = json_output.split("```")[0].strip()
                return output

        return json_output.strip()

    def _load_and_resize_image(self, image_path: str) -> Image.Image:
        """
        載入並調整圖像大小

        Args:
            image_path: 圖像檔案路徑

        Returns:
            調整大小後的圖像
        """
        try:
            image = Image.open(image_path)
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            logger.debug(f"已載入圖像：{image_path}，大小：{image.size}")
            return image
        except Exception as e:
            logger.error(f"載入圖像失敗：{e}")
            raise ValidationError(f"無法載入圖像：{str(e)}")

    def detect_objects(self, image_path: str, target_item: str) -> List[DetectedObject]:
        """
        偵測圖像中的物件

        Args:
            image_path: 圖像檔案路徑
            target_item: 目標物件描述

        Returns:
            偵測到的物件列表
        """
        try:
            logger.info(f"開始偵測圖像中的物件：{target_item}")

            image = self._load_and_resize_image(image_path)

            prompt = f"""Please analyze this image and detect all instances of {target_item}.

                For each detected object, provide:
                1. A bounding box in the format: [y_min, x_min, y_max, x_max] with values from 0 to 1000 (normalized coordinates)
                2. A descriptive label for the object

                IMPORTANT: You MUST respond with ONLY a valid JSON array, nothing else. No markdown, no explanations, just pure JSON.

                Example format:
                [{{"box_2d": [100, 150, 300, 400], "label": "dog in center"}}, {{"box_2d": [50, 50, 200, 200], "label": "cat on left"}}]

                If no {target_item} is found in the image, return an empty array: []

                Now analyze the image and provide the JSON response:
                """

            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.1,  # Lower temperature for more deterministic responses
            )

            logger.debug("正在呼叫 Gemini API 進行物件偵測")
            response = self.client.models.generate_content(
                model=self.model, contents=[prompt, image], config=config
            )

            logger.debug(f"API 回應類型：{type(response)}")
            logger.debug(f"API 回應內容：{response}")

            # 嘗試獲取回應文本
            if hasattr(response, "text") and response.text:
                response_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                # 從候選對象中提取文本
                response_text = (
                    response.candidates[0].content.parts[0].text
                    if response.candidates[0].content.parts
                    else ""
                )
            else:
                raise ValidationError("Gemini API 沒有返回有效的回應文本")

            logger.debug(
                f"提取的回應文本：{response_text[:200] if response_text else '空'}"
            )

            parsed_json = self._parse_json_response(response_text)

            if not parsed_json or not parsed_json.strip():
                logger.warning(f"API 回應為空，返回空的偵測結果")
                return []

            # 嘗試解析 JSON，如果失敗則嘗試提取 JSON 數組
            try:
                items = json.loads(parsed_json)
            except json.JSONDecodeError:
                # 嘗試從回應中提取 JSON 數組
                logger.debug("直接 JSON 解析失敗，嘗試提取 JSON 數組...")
                import re

                json_match = re.search(r"\[.*\]", parsed_json, re.DOTALL)
                if json_match:
                    try:
                        items = json.loads(json_match.group())
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"無法從回應中提取有效的 JSON：{parsed_json[:300]}"
                        )
                        raise ValidationError(
                            f"API 回應不是有效的 JSON 格式。回應內容：{parsed_json[:200]}"
                        )
                else:
                    logger.error(f"未找到 JSON 數組，回應內容：{parsed_json[:300]}")
                    raise ValidationError(
                        f"API 回應中未找到 JSON 數組。回應內容：{parsed_json[:200]}"
                    )

            # 如果 items 不是列表，嘗試將其轉換為列表
            if not isinstance(items, list):
                logger.warning(f"API 回應不是列表，嘗試轉換...")
                items = [items] if items else []

            detected_objects = [
                DetectedObject(box_2d=item["box_2d"], label=item["label"])
                for item in items
                if isinstance(item, dict) and "box_2d" in item and "label" in item
            ]

            logger.info(f"成功偵測到 {len(detected_objects)} 個物件")
            return detected_objects

        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析錯誤：{e}")
            logger.error(
                f"回應內容：{parsed_json[:500] if 'parsed_json' in locals() else 'N/A'}"
            )
            raise ValidationError(
                f"API 回應格式錯誤：{str(e)}。回應內容：{parsed_json[:200] if 'parsed_json' in locals() else 'N/A'}"
            )
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"物件偵測失敗：{e}")
            raise

    def _draw_bounding_box(
        self,
        image: Image.Image,
        detected_object: DetectedObject,
        box_color: Tuple[int, int, int, int] = (255, 0, 0, 255),
        box_width: int = 5,
    ) -> Image.Image:
        """
        在圖像上繪製邊界框和標籤

        Args:
            image: 原始圖像
            detected_object: 偵測到的物件
            box_color: 邊界框顏色 (R, G, B, A)
            box_width: 邊界框寬度

        Returns:
            標註後的圖像
        """
        box = detected_object.box_2d
        y0 = int(box[0] / 1000 * image.size[1])
        x0 = int(box[1] / 1000 * image.size[0])
        y1 = int(box[2] / 1000 * image.size[1])
        x1 = int(box[3] / 1000 * image.size[0])

        if y0 >= y1 or x0 >= x1:
            logger.warning(f"無效的邊界框：{box}")
            return None

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=box_width)

        label_text = detected_object.label
        text_y = max(5, y0 - 25)
        draw.rectangle(
            [x0, text_y, x0 + len(label_text) * 10, text_y + 20],
            fill=(box_color[0], box_color[1], box_color[2], 200),
        )
        draw.text((x0 + 5, text_y + 5), label_text, fill=(255, 255, 255, 255))

        composite = Image.alpha_composite(image.convert("RGBA"), overlay)
        return composite

    def annotate_image(
        self, image_path: str, target_item: str, output_dir: str = None
    ) -> List[str]:
        """
        對圖像進行標註並儲存結果

        Args:
            image_path: 輸入圖像路徑
            target_item: 目標物件描述
            output_dir: 輸出目錄路徑

        Returns:
            儲存的檔案路徑列表
        """
        try:
            logger.info(f"開始標註圖像：{image_path}")

            if output_dir is None:
                settings = get_settings()
                output_dir = str(settings.output_dir / "Annotated_Image")

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"輸出目錄：{output_dir}")

            detected_objects = self.detect_objects(image_path, target_item)

            if not detected_objects:
                logger.warning("未偵測到任何物件")
                return []

            image = self._load_and_resize_image(image_path)

            saved_files = []
            for i, obj in enumerate(detected_objects):
                logger.debug(f"正在處理物件 {i+1}/{len(detected_objects)}：{obj.label}")

                annotated = self._draw_bounding_box(image, obj)

                if annotated is None:
                    continue

                output_filename = f"{obj.label.replace(' ', '_')}_{i}_detected.png"
                output_file = output_path / output_filename
                annotated.save(str(output_file))

                saved_files.append(str(output_file))
                logger.debug(f"✓ 已儲存：{output_filename}")

            logger.info(f"標註完成，共儲存 {len(saved_files)} 個檔案")
            return saved_files

        except Exception as e:
            logger.error(f"圖像標註失敗：{e}")
            raise

    def get_detection_summary(self, image_path: str, target_item: str) -> Dict:
        """
        獲取偵測摘要資訊

        Args:
            image_path: 圖像檔案路徑
            target_item: 目標物件描述

        Returns:
            包含偵測資訊的字典
        """
        try:
            detected_objects = self.detect_objects(image_path, target_item)

            return {
                "image_path": image_path,
                "target_item": target_item,
                "total_detected": len(detected_objects),
                "objects": [obj.to_dict() for obj in detected_objects],
            }
        except Exception as e:
            logger.error(f"獲取偵測摘要失敗：{e}")
            raise
