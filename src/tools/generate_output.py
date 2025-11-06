import openai
import os
import sys

def generate_output(prompt, model="gpt-4.1-nano-2025-04-14", temperature=0.7, max_tokens=1000):
    """
    使用 OpenAI 的 GPT 模型生成回應。
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("請設定 OPENAI_API_KEY 環境變數。")
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是個優秀的助理，幫我將下列的資訊整理成有條理的回答。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error generating output: {e}", file=sys.stderr)
        return None