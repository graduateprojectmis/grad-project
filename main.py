from tools.airpods_manual_fetch import scrape_airpods_manual
from tools.generate_embedding_gemini import process_and_embed_data, process_and_embed_questions
from tools.load_save_data import load_json_data, save_to_json
from tools.similarity_calculation import calculate
from tools.similarity_calculation import print_similarity_results
import os

def main():
    """
    # Fetch and embed AirPods manual data
    url = "https://support.apple.com/en-us/guide/airpods/welcome/web"
    input_data = scrape_airpods_manual(url)
    """
    embedding_output_file = "output/json/text_embedding_gemini.json"
    question_output_file = "output/json/question_embeddings.json"
    """
    if input_data:
        final_data_with_embeddings = process_and_embed_data(input_data)
        save_to_json(final_data_with_embeddings, embedding_output_file)
    """
    # Create and embed questions
    questions = [
        "How to pair AirPods with an iPhone?",
        "How to reset AirPods?",
        "How to check AirPods battery status?",
        "How to use AirPods with a Mac?",
    ]

    if questions:
        question_embeddings = process_and_embed_questions(questions)
        save_to_json(question_embeddings, question_output_file)
    
    # Calculate similarity and display results
    qa_data = calculate(question_output_file, embedding_output_file)
    print_similarity_results(qa_data)
    save_to_json(qa_data, "output/json/similarity_results.json")

if __name__ == '__main__':
    main()