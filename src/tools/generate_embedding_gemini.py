import json
import google.generativeai as genai
import os
import time
import dotenv

Model_Name = 'models/text-embedding-004'

dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def process_and_embed_data(raw_data: list, model_name: str = Model_Name) -> list:
    """
    Process raw data and batch generate titles and chunks in each title。

    Args:
        raw_data: List that fetched from JSON file。
        model_name: Which model we use。

    Returns:
        A List that contain title, chunks, embeddings。
        format:
        [
            {
                "title": "Title text",
                "title_embedding": [...],
                "chunks": [
                    {
                        "chunk_text": "Chunk text",
                        "chunk_embedding": [...]
                    },
                    ...
                ]
            },
            ...
        ]
    """
    if not raw_data:
        return []

    texts_to_embed = []
    processed_data = []

    print("Getting data and preparing for embedding...")
    

    for item in raw_data:
        title = item.get('title', '')
        content = item.get('content', '')

        # Add title to texts_to_embed if it exists
        if title:    
            texts_to_embed.append(title)
        
        """
        Should find a better way to split content into chunks.
        """
        chunks = [chunk.strip() for chunk in content.split('\n') if chunk.strip()]
        
        # Add chunks to texts_to_embed
        texts_to_embed.extend(chunks)

        # Prepare processed_data structure
        processed_data.append({
            "title": title,
            "title_embedding": [],
            "chunks": [{"chunk_text": chunk, "chunk_embedding": []} for chunk in chunks]
        })

    print(f"Chunks waiting for batch embedding：{len(texts_to_embed)}")

    print("Batch Embedding...")
    try:
        
        # Gemini api only support max 100 texts per request so we set batch_size = 100.
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i+batch_size]
            
            # To avoid hitting the per-minute request limit, we can embed in batches with a short delay between batches.
            if i > 0:
                print(f"{i} items has been processed, Wait 1 sec...")
                time.sleep(1)
            
            result = genai.embed_content(model=model_name,
                                         content=batch,
                                         task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(result['embedding'])
            
    except Exception as e:
        print(f"Error occurs when calling API : {e}")
        return []

    print("Mapping data to processed_data that initialized earlier...")
    embedding_idx = 0
    for doc in processed_data:
        # Corresponding title's embedding
        doc['title_embedding'] = all_embeddings[embedding_idx]
        embedding_idx += 1
        
        # Corresponding chunks' embeddings
        for chunk in doc['chunks']:
            chunk['chunk_embedding'] = all_embeddings[embedding_idx]
            embedding_idx += 1
            
    print("Data processing and embedding completed.")
    return processed_data

def process_and_embed_questions(questions: list, model_name: str = Model_Name) -> list:
    """
    Process raw data and batch generate questions。

    Args:
        raw_data: List that fetched from main.py pass。
        model_name: Which model we use。

    Returns:
        A List that contain title, chunks, embeddings。
        format:
        [
            {
                "question": "Question text",
                "question_embedding": [...]
            },
            ...
        ]
    """
    if not questions:
        return []

    print("Prepare content for embedding...")
    try:
        result = genai.embed_content(model=model_name,
                                     content=questions,
                                     task_type="RETRIEVAL_QUERY")
        embeddings = result['embedding']
    except Exception as e:
        print(f"Error occurs when calling API : {e}")
        return []

    question_data = []
    for question, embedding in zip(questions, embeddings):
        question_data.append({
            "question": question,
            "question_embedding": embedding
        })

    print("Question embedding completed.")
    return question_data
