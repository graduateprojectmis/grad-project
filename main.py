from tools.airpods_manual_fetch import scrape_airpods_manual
from tools.generate_embedding_gemini import process_and_embed_data, process_and_embed_questions
from tools.load_save_data import load_json_data, save_to_json
from tools.similarity_calculation import calculate
import os

def main():
    # Fetch and embed AirPods manual data
    url = "https://support.apple.com/en-us/guide/airpods/welcome/web"
    # output_file = "../output/json/airpods_manual_data.json"
    input_data = scrape_airpods_manual(url)
    
    embedding_output_file = "../output/json/text_embedding_gemini.json"
    
    if input_data:
        final_data_with_embeddings = process_and_embed_data(input_data)
        save_to_json(final_data_with_embeddings, embedding_output_file)
        
    # Create and embed questions
    questions = [
        "How to pair AirPods with an iPhone?",
        "How to reset AirPods?",
        "How to check AirPods battery status?",
        "How to use AirPods with a Mac?",
    ]
    question_embeddings = process_and_embed_questions(questions)
    
    question_output_file = "../output/json/question_embeddings.json"
    save_to_json(question_embeddings, question_output_file)
    
    # Calculate similarity and display results
    calculate()

if __name__ == '__main__':
    main()