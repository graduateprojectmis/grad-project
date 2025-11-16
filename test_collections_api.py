"""
測試 Collections API 的腳本
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_get_collections():
    """測試獲取 collections 列表"""
    print("=" * 60)
    print("測試: GET /api/collections")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/api/collections")
        print(f"狀態碼: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"當前 collection: {data['current_collection']}")
            print(f"可用的 collections:")
            for col in data["collections"]:
                print(f"  - {col['name']}: {col['count']} 條文件")
            return data
        else:
            print(f"錯誤: {response.text}")
            return None
    except Exception as e:
        print(f"連接失敗: {e}")
        return None


def test_switch_collection(collection_name):
    """測試切換 collection"""
    print("\n" + "=" * 60)
    print(f"測試: POST /api/switch-collection (切換到 {collection_name})")
    print("=" * 60)

    try:
        response = requests.post(
            f"{BASE_URL}/api/switch-collection",
            json={"collection_name": collection_name},
        )
        print(f"狀態碼: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"訊息: {data['message']}")
            print(f"當前 collection: {data['current_collection']}")
            print(f"文件數量: {data['document_count']}")
            return data
        else:
            print(f"錯誤: {response.text}")
            return None
    except Exception as e:
        print(f"連接失敗: {e}")
        return None


def test_health_after_switch():
    """測試切換後的健康檢查"""
    print("\n" + "=" * 60)
    print("測試: GET /api/health (切換後)")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"狀態碼: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"狀態: {data['status']}")
            print(f"資料庫文件數量: {data['chroma_db_count']}")
            return data
        else:
            print(f"錯誤: {response.text}")
            return None
    except Exception as e:
        print(f"連接失敗: {e}")
        return None


if __name__ == "__main__":
    print("\n🚀 開始測試 Collections API")
    print("請確保後端服務正在運行 (python run_api.py)")
    print()

    # 測試 1: 獲取 collections 列表
    collections_data = test_get_collections()

    if collections_data:
        # 測試 2: 切換 collection（如果有多個的話）
        current = collections_data["current_collection"]
        collections = collections_data["collections"]

        if len(collections) > 1:
            # 找一個不同的 collection 來切換
            other_collection = None
            for col in collections:
                if col["name"] != current:
                    other_collection = col["name"]
                    break

            if other_collection:
                test_switch_collection(other_collection)
                test_health_after_switch()

                # 切回原來的 collection
                print("\n" + "=" * 60)
                print(f"切回原來的 collection: {current}")
                print("=" * 60)
                test_switch_collection(current)
        else:
            print("\n⚠️ 只有一個 collection，無法測試切換功能")

    print("\n✅ 測試完成！")
