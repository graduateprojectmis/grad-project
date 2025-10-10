from langchain_text_splitters import RecursiveCharacterTextSplitter
from .clean_data import preprocess_text
import os
import json
import openai
import dotenv

Model_Name = 'text-embedding-3-small'

dotenv.load_dotenv()

class EmbeddingGenerator:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_embedding(self, chunks):
        '''
        embed text using OpenAI API
        Args:
            chunks: list of text chunks
        Returns:
            list of embeddings
        '''
        response = openai.Embedding.create(
            model=Model_Name,
            input=chunks
        )
        return [item["embedding"] for item in response["data"]]

# 初始化 EmbeddingGenerator
embedding_generator = EmbeddingGenerator(api_key=os.getenv("OPENAI_API_KEY"))

def split_text(content, chunk_size=600, chunk_overlap=30):
    '''
    split text into chunks
    Args:
        content: raw text
        chunk_size: max length of each chunk
        chunk_overlap: overlap length between chunks
    Returns:
        list of chunks
    '''
    # 初始化文本切分器，設定分割規則
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["。", "！", "？", "\n", "，", " "]
    )

    # 清理原始文字
    cleaned_text = preprocess_text(content)
    # 切分成 chunk
    chunks = text_splitter.split_text(cleaned_text)

    return chunks

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
        
        chunks = split_text(content)
        
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
        all_embeddings = embedding_generator.generate_embedding(texts_to_embed)
            
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
        embeddings = embedding_generator.generate_embedding(questions)
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
