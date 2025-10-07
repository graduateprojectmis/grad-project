from tools.airpods_manual_fetch import scrape_airpods_manual
from tools.generate_embedding_gemini import process_and_embed_data, process_and_embed_questions
from tools.load_save_data import load_json_data, save_to_json
import os

def main():
    """
    # Step 1: 爬取 AirPods 使用手冊並儲存為 JSON
    url = "https://support.apple.com/zh-tw/guide/airpods/welcome/web"
    # output_file = "output/json/airpods_manual_data.json"
    input_data = scrape_airpods_manual(url)
    
    # Step 2: 產生並儲存 Embeddings
    embedding_output_file = "output/json/text_embedding_gemini.json"
    
    if input_data:
        final_data_with_embeddings = process_and_embed_data(input_data)
        save_to_json(final_data_with_embeddings, embedding_output_file)
    
    """
    
    # Step 3: 處理並產生問題的 Embeddings
    questions = [
        "AirPods 的電池續航力有多長？",
        "如何將 AirPods 與裝置配對？",
        "AirPods 支援哪些語音助理？"
    ]
    question_embeddings = process_and_embed_questions(questions)
    
    question_output_file = "output/json/question_embeddings.json"
    save_to_json(question_embeddings, question_output_file)

if __name__ == '__main__':
    main()