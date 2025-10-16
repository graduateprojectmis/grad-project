from langchain_text_splitters import RecursiveCharacterTextSplitter
from .clean_data import preprocess_text
from .load_save_data import load_json_data, save_to_json
import os
import json
from openai import OpenAI
import dotenv
import base64

Model_Name = 'text-embedding-3-small'

dotenv.load_dotenv()

class ImageProcessor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    # Helper function to convert a file to base64 representation
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Takes in a base64 encoded image and prompt (requesting an image summary)
    # Returns a response from the LLM (image summary)
    def image_summarize(self, img_base64, prompt):
        '''
        summarize image using OpenAI API
        Args:
            img_path: path to image file(str)
        Returns:
            image description(str)
        '''
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}",
                        },
                        },
                    ],
                    }
                ],
                max_tokens=150,
            )
            content = response.choices[0].message.content
            print(f"    [Info] Image summarized: {content}")
            return content
        except Exception as e:
            print(f"Error occurs when summarizing image: {e}.")
            return "No description available."

class EmbeddingGenerator:
    def __init__(self, api_key: str):
        # 使用新版 OpenAI SDK
        self.client = OpenAI(api_key=api_key)

    def generate_embedding(self, chunks):
        '''
        embed text using OpenAI API
        Args:
            chunks: list of text chunks
        Returns:
            list of embeddings
        '''
        response = self.client.embeddings.create(
            model=Model_Name,
            input=chunks
        )
        return [item.embedding for item in response.data]
    
    def generate_image_embedding(self, image_paths):
        '''
        embed images using OpenAI API
        Args:
            image_paths: list of image file paths
        Returns:
            list of dict with image url, description, and embedding
        '''
        image_processor = ImageProcessor(api_key=os.getenv("OPENAI_API_KEY"))
        embedded_images = []
        
        for img_path in image_paths:
            if not os.path.isfile(img_path):
                print(f"    [Warning] Image file {img_path} does not exist. Skipping.")
                continue
            
            try:
                img_base64 = image_processor.encode_image(img_path)
                img_description = image_processor.image_summarize(
                    img_base64, 
                    prompt="請用中文描述這張圖片的內容，50字以內，圖片主要是關於蘋果耳機。"
                )
                
                embedding = self.generate_embedding(img_description)
                embedded_images.append({
                    "img_url": img_path,
                    "img_description": img_description,
                    "img_embedding": embedding[0] if embedding else []
                })
            except Exception as e:
                print(f"    [Error] Failed to process image {img_path}: {e}")
                continue
        
        return embedded_images

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
                "images": [
                    {
                        "img_url": "Image URL",
                        "img_description": "Image description",
                        "img_embedding": [...]
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
    

    start = 1
    images = os.listdir('images')

    for item in raw_data:
        title = item.get('title', '')
        content = item.get('content', '')
        # Get image filenames from local 'images' folder, sorted
        curr_images = item.get('images', [])
        images_size = len(curr_images)
    

        # Add title to texts_to_embed if it exists
        if title:    
            texts_to_embed.append(title)
        
        chunks = split_text(content)
        
        # Add chunks to texts_to_embed
        texts_to_embed.extend(chunks)

        # Prepare processed_data structure
        processed_item = {
            "title": title,
            "title_embedding": [],
            "chunks": [{"chunk_text": chunk, "chunk_embedding": []} for chunk in chunks],
            "images": []
        }
        if images:
            for i in range(start, start + images_size):
                img_path = os.path.join('images', f"image{i}.jpg")
                if os.path.isfile(img_path):
                    processed_item["images"].append(img_path)
            start += images_size
        processed_data.append(processed_item)

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

        # Corresponding images' embeddings
        if doc['images']:
            embedded_images = embedding_generator.generate_image_embedding(doc['images'])
            doc['images'] = embedded_images
        else:
            doc['images'] = []
            
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

if __name__ == '__main__':
    input_filename = 'output/json/text_and_image_airpods_manual_data.json'
    output_filename = 'output/json/text_and_image_embedding_openai.json'

    raw_data = load_json_data(input_filename)
    if not raw_data:
        print(f"Failed to load data from {input_filename}. Exiting.")
        exit(1)
    embedded_data = process_and_embed_data(raw_data, model_name=Model_Name)
    try:
        save_to_json(embedded_data, output_filename)
        print(f"Data successfully saved to {output_filename}.")
    except Exception as e:
        print(f"Failed to save data to {output_filename}: {e}")