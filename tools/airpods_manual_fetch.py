"""
1. 透過抓取Airpods Support網站的toc hasIcons類別獲得目錄
2. 便利每個分頁並且獲取其中的AppleTopic apd-topic dark-mode-enabled book book-content類別中的文字跟圖片
3. 輸出結果爲JSON檔

提供兩種使用方法：
    1. 傳入url與output_filename參數，會將結果儲存至指定檔案
    2. 僅傳入url參數，會回傳結果列表一個list[dict]
"""

import requests
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin
from .load_save_data import load_json_data, save_to_json

def scrape_airpods_manual(url: str, output_filename = "") -> list:
    # 儲存已下載的圖片URL，避免重複下載
    seen = set()
    image_folder = "images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_counter = 1

    toc_url = url
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"正在抓取目錄頁面：{toc_url}")

    try:
        # 獲取目錄頁面
        response = requests.get(toc_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 找到目錄列表
        toc_list = soup.select_one('ul.toc.hasIcons')
        
        if not toc_list:
            print("錯誤：找不到指定的目錄列表 (class='toc hasIcons')")
            return None
            
        page_links = toc_list.find_all('a')
        
        rag_database = []
        
        print(f"找到 {len(page_links)} 個說明頁面連結。抓取內容...")
        
        # 遍歷所有連結，抓取分頁內容
        for i, link in enumerate(page_links):
            page_title = link.get_text(strip=True)
            relative_url = link.get('href')
            page_url = urljoin(toc_url, relative_url)
            
            print(f"  ({i+1}/{len(page_links)}) 正在處理: {page_title} - {page_url}")
            
            try:
                # 抓取每個說明的詳細內容
                page_response = requests.get(page_url, headers=headers)
                page_response.raise_for_status()
                
                page_soup = BeautifulSoup(page_response.text, 'html.parser')
                
                content_div = page_soup.find('div', class_='AppleTopic apd-topic dark-mode-enabled book book-content')
                
                if content_div:
                    # fetch and clean text
                    content_text = content_div.get_text(separator='\n', strip=True)
                    
                    # fetch images
                    images = []
                    for img_tag in content_div.find_all('img'):
                        img_src = img_tag.get('src')
                        img_alt = img_tag.get('alt', '')
                        if img_src:
                            full_img_url = urljoin(page_url, img_src)
                            if full_img_url not in seen:
                                images.append({'url': full_img_url, 'alt': img_alt})
                                seen.add(full_img_url)

                    # Download images to local folder(images/)
                    for img in images:
                        try:
                            img_data = requests.get(img['url'], headers=headers)
                            img_data.raise_for_status()
                            img_filename = f"image{image_counter}.jpg"
                            img_path = os.path.join(image_folder, img_filename)
                            with open(img_path, 'wb') as f:
                                f.write(img_data.content)
                            image_counter += 1
                        except requests.RequestException as e:
                            print(f"    [Error] 下載圖片 {img['url']} 時發生錯誤: {e}")

                    rag_database.append({
                        'title': page_title,
                        'url': page_url,
                        'content': content_text,
                        'images': images
                    })
                else:
                    print(f"    [Warning] 在頁面 '{page_title}' 中找不到 class='AppleTopic apd-topic dark-mode-enabled book book-content' 的內容區塊")

                time.sleep(0.5)

            except requests.RequestException as e:
                print(f"    [Error] 抓取頁面 '{page_title}' ({page_url}) 時發生錯誤: {e}")

        if output_filename:
            save_to_json(rag_database, output_filename)
        else:
            return rag_database

    except requests.RequestException as e:
        print(f"無法訪問目錄頁面 {toc_url}。錯誤: {e}")
        return None

if __name__ == '__main__':
    data = scrape_airpods_manual("https://support.apple.com/zh-tw/guide/airpods/welcome/web", "output/json/text_and_image_airpods_manual_data.json")
