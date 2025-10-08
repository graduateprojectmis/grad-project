from tools.airpods_manual_fetch import scrape_airpods_manual
from tools.load_save_data import load_json_data, save_to_json
from tools.similarity_calculation import calculate
from tools.similarity_calculation import print_similarity_results
import os

def main():
    LLM = "openai"
    
    try:
        import importlib
        embedding_module = importlib.import_module(f"tools.generate_embedding_{LLM}")
        process_and_embed_data = embedding_module.process_and_embed_data
        process_and_embed_questions = embedding_module.process_and_embed_questions
    except ImportError:
        print(f"{LLM} embedding module not found")
    """
    # Fetch and embed AirPods manual data
    url = "https://support.apple.com/en-us/guide/airpods/welcome/web"
    input_data = scrape_airpods_manual(url)
    """
    embedding_output_file = f"output/json/text_embedding_{LLM}.json"
    question_output_file = f"output/json/question_embeddings_{LLM}.json"
    """
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

    if not questions:
        question_embeddings = process_and_embed_questions(questions)
        save_to_json(question_embeddings, question_output_file)
    """
    # Calculate similarity and display results
    qa_data = calculate(
        question_file=question_output_file,
        data_file=embedding_output_file,
        chunk_top_percentage=0.5
        )
    print_similarity_results(qa_data)
    save_to_json(qa_data, f"output/json/similarity_results_{LLM}.json")

if __name__ == '__main__':
    main()