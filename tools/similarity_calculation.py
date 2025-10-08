"""
Module for calculating cosine similarity between vectors and finding the most similar content.
Steps:
1. Load data with embeddings and questions with embeddings from JSON files.
2. For each question, compute cosine similarity with titles and return top_k (Default is 5) similar titles.
3. From the top similar titles, extract their chunks and compute cosine similarity with the question embedding.
4. Return the top_k most similar chunks for each question.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
from numpy.typing import NDArray

from .load_save_data import load_json_data


def calculate_cosine_similarity(
    vec1: Union[List[float], NDArray[np.float64]], 
    vec2: Union[List[float], NDArray[np.float64]]
) -> float:
    
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1: First vector (list or numpy array)
        vec2: Second vector (list or numpy array)
        
    Returns:
        float: Cosine similarity value ranging from -1 to 1
        
    Raises:
        ValueError: If the dimensions of the vectors do not match
    """
    
    vec1_array = np.array(vec1, dtype=np.float64)
    vec2_array = np.array(vec2, dtype=np.float64)
    
    if vec1_array.shape != vec2_array.shape:
        raise ValueError(f"Dimension do not match: {vec1_array.shape} vs {vec2_array.shape}")

    # Calculate norms
    norm_vec1 = np.linalg.norm(vec1_array)
    norm_vec2 = np.linalg.norm(vec2_array)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    # Calculate cosine similarity
    dot_product = np.dot(vec1_array, vec2_array)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return float(similarity)



def _calculate_similarities_for_items(
    query_embedding: NDArray[np.float64],
    items: List[Dict[str, Any]],
    embedding_key: str,
    text_key: str,
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    
    """
    Calculate similarities between a query embedding and a list of items.
    
    Args:
        query_embedding: The embedding vector for the query
        items: List of items to compare against
        embedding_key: The key in each item dict that contains the embedding
        text_key: The key in each item dict that contains the text
        
    Returns:
        List[Dict]: A list of dicts containing the text and similarity score
    """

    similarities = []
    
    for item in items:
        embedding = item.get(embedding_key)
        text = item.get(text_key)
        
        if embedding is not None and text is not None:
            try:
                similarity = calculate_cosine_similarity(query_embedding, embedding)
                similarities.append({
                    text_key: text,
                    'similarity': similarity
                })
            except (ValueError, Exception) as e:
                print(f"Warning : Calculate cosine similarity Error - {e}")
                continue
    
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities if top_k is None else similarities[:top_k]


def find_most_similar_chunks(
    query_embedding: Union[List[float], NDArray[np.float64]], 
    data_with_embeddings: List[Dict[str, Any]], 
    title_top_k: int = 5,
    chunk_top_k: int = 5,
    include_titles: bool = True
) -> List[Dict[str, Any]]:
    
    """
    Find the top K most similar chunks in the dataset to the query vector.
    
    Args:
        query_embedding: The embedding vector for the query (list or numpy array)
        data_with_embeddings: List of data items containing chunks and their embeddings
        top_k: Number of top similar chunks to return (default is 5)
        include_titles: Whether to include title similarity in the calculation (default is True)
        
    Returns:
        List[Dict]: A list of the top K most similar items, each dict contains text and similarity score
        
    Raises:
        ValueError: If the input data is invalid
    """
    
    if not data_with_embeddings:
        raise ValueError("Data with embeddings is empty or None")
    
    # Convert query embedding to numpy array
    query_array = np.array(query_embedding, dtype=np.float64)
    
    all_similarities = []
    
    # Calculate title similarities if required
    if include_titles:
        title_similarities = _calculate_similarities_for_items(
            query_array,
            data_with_embeddings,
            'title_embedding',
            'title',
            title_top_k
        )
        
        # Find the items corresponding to the most similar titles
        top_titles = [item['title'] for item in title_similarities]
        print(f"Top similar titles: {top_titles}")
        
        # Filter data items that match the top similar titles
        relevant_items = [
            item for item in data_with_embeddings 
            if item.get('title') in top_titles
        ]
        print(f"Found {len(relevant_items)} items with matching titles")
        
        # Extract chunks only from the relevant items (titles with high similarity)
        chunk_list = []
        for item in relevant_items:
            chunks = item.get('chunks', [])
            chunk_list.extend(chunks)
    else:
        # If not including titles, search all chunks
        chunk_list = []
        for item in data_with_embeddings:
            chunks = item.get('chunks', [])
            chunk_list.extend(chunks)
    
    # Calculate chunk similarities from the filtered chunk list
    chunk_similarities = _calculate_similarities_for_items(
        query_array,
        chunk_list,
        'chunk_embedding',
        'chunk_text',
        chunk_top_k  # Get more results to ensure we have enough after deduplication
    )
    all_similarities.extend(chunk_similarities)
    
    # According to similarity sort and return top_k
    all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Remove duplicates based on chunk_text while preserving order
    seen_texts = set()
    unique_similarities = []
    for item in all_similarities:
        text = item.get('chunk_text', '')
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_similarities.append(item)
    
    return unique_similarities[:chunk_top_k]


def process_questions_similarity(
    questions_with_embeddings: List[Dict[str, Any]],
    data_with_embeddings: List[Dict[str, Any]],
    title_top_k: int = 5,
    chunk_top_k: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    
    """
    Process multiple questions to find the most similar content for each question.
    
    Args:
        questions_with_embeddings: List of questions with their embeddings
        data_with_embeddings: List of data items with their embeddings
        top_k: Number of top similar items to return for each question (default is 5
        
    Returns:
        Dict: A mapping of questions to their most similar content
    """
    
    results = {}
    
    for question_item in questions_with_embeddings:
        question = question_item.get('question')
        question_embedding = question_item.get('question_embedding')
        
        if not question or question_embedding is None:
            print(f"Warning: Question or its embedding is missing, skipping...")
            continue
        
        try:
            top_similar_chunks = find_most_similar_chunks(
                question_embedding, 
                data_with_embeddings, 
                title_top_k,
                chunk_top_k,
            )
            results[question] = top_similar_chunks
        except Exception as e:
            print(f"Error: Error occurs when process question : '{question}' - {e}")
            continue
    
    return results


def print_similarity_results(
    results: Dict[str, List[Dict[str, Any]]],
    max_text_length: int = 200
) -> None:
    
    """
    Print the similarity calculation results.
    
    Args:
        results: Mapping of questions to their similar content
        max_text_length: Maximum length of text to display (default is 100 characters)
    """
    
    for question, similar_chunks in results.items():
        print(f"\n{'='*80}")
        print(f"Question : {question}")
        print(f"{'='*80}")
        print("Most relative chunks :")
        
        for idx, chunk in enumerate(similar_chunks, 1):
            # Fetch text content
            text = chunk.get('chunk_text') or chunk.get('title_text', '')
            similarity = chunk['similarity']
            
            # Only display part of the text
            display_text = text[:max_text_length]
            if len(text) > max_text_length:
                display_text += "..."
            
            print(f"\n  {idx}. Similarity : {similarity:.4f}")
            print(f"     Content : {display_text}")
        print()


def calculate(question_file: str, data_file:str) -> None:
    """
    Main function: Load data, calculate similarities, and display results.
    """
    
    try:
        print("Loading JSON file...")
        data_with_embeddings = load_json_data(data_file)
        questions_with_embeddings = load_json_data(question_file)
        
        if not data_with_embeddings:
            print(f"Error : Can't load file in {data_file} ")
            return
        
        if not questions_with_embeddings:
            print(f"Error : Can't load file in {question_file} ")
            return
        
        print(f"Successfully loaded {len(data_with_embeddings)} data entries")
        print(f"Successfully loaded {len(questions_with_embeddings)} questions")
        
        print("\nStart similarity calculation...")
        results = process_questions_similarity(
            questions_with_embeddings,
            data_with_embeddings,
            title_top_k=5,
            chunk_top_k=10
        )
        
        print_similarity_results(results)
        
        print(f"\nSuccessfully processed {len(results)} questions")
        
    except FileNotFoundError as e:
        print(f"Error : File not found - {e}")
    except Exception as e:
        print(f"Error : An unexpected error occurred during execution - {e}")
        import traceback
        traceback.print_exc()
