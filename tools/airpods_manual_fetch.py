"""
1. 透過抓取Airpods Support網站的toc hasIcons類別獲得目錄
2. 便利每個分頁並且獲取其中的AppleTopic apd-topic dark-mode-enabled book book-content類別中的文字
3. 輸出結果爲JSON檔
"""

import requests
import json
from bs4 import BeautifulSoup
import time
import os
from urllib.parse import urljoin

def scrape_airpods_manual():
    toc_url = "https://support.apple.com/zh-tw/guide/airpods/welcome/web"
    
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
                    content_text = content_div.get_text(separator='\n', strip=True)
                    
                    rag_database.append({
                        'title': page_title,
                        'url': page_url,
                        'content': content_text
                    })
                else:
                    print(f"    [Warning] 在頁面 '{page_title}' 中找不到 class='AppleTopic apd-topic dark-mode-enabled book book-content' 的內容區塊")

                time.sleep(0.5)

            except requests.RequestException as e:
                print(f"    [Error] 抓取頁面 '{page_title}' ({page_url}) 時發生錯誤: {e}")

        return rag_database

    except requests.RequestException as e:
        print(f"無法訪問目錄頁面 {toc_url}。錯誤: {e}")
        return None

if __name__ == '__main__':
    airpods_data = scrape_airpods_manual()
    
    if airpods_data:
        output_filename = 'output/json/airpods_manual_data.json'
        
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(airpods_data, f, ensure_ascii=False, indent=4)
        
        print(f"[Function Ended]資料已儲存至檔案：{output_filename}")
