import langid
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download("punkt")
nltk.download("stopwords")


def extract_keywords(text) -> list:
    """
    Extract keywords from text in Chinese or English using language-specific processing.

    This function automatically detects the language of the input text and applies
    appropriate keyword extraction methods:
    - For Chinese: Uses jieba segmentation with custom dictionary support
    - For English: Uses NLTK tokenization with stopword filtering

    Args:
        text (str): The input text from which to extract keywords.

    Returns:
        list[str]: A list of extracted keywords. For Chinese text, returns
                  segmented words excluding punctuation. For English text,
                  returns lowercase alphabetic words excluding stopwords.

    Example:
        >>> extract_keywords("My AirPods Pro suddenly has no sound, and the left earphone is completely silent. Is the battery dead? Does it need to be repaired?")
        ['airpods', 'pro', 'suddenly', 'sound', 'left', 'earphone', 'completely', 'silent', 'battery', 'dead', 'need', 'repaired']

        >>> extract_keywords("我的AirPods Pro突然沒有聲音了，左邊耳機完全聽不到，是不是電池壞了？需要拿去維修嗎？")
        ['我', '的', 'AirPods', 'Pro', '突然', '沒有', '聲音', '了', '左邊', '耳機', '完全', '聽', '不到', '是不是', '電池', '壞', '了', '需要', '拿', '去', '維修', '嗎']
    """
    keywords = []
    lang = langid.classify(text)[0]

    if lang == "zh":
        chinese_punctuation = set("，。！？；：\"'（）【】《》〈〉—…、「」")

        dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir, "extract_keywords_dict.txt")
        if os.path.exists(file_path):
            jieba.load_userdict(file_path)

        # Use jieba to extract keywords
        raw_keywords = list(jieba.cut(text, cut_all=False))
        keywords = [
            word
            for word in raw_keywords
            if word.strip() and word not in chinese_punctuation
        ]

    elif lang == "en":
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text)
        keywords = [
            w.lower() for w in words if w.isalpha() and w.lower() not in stop_words
        ]

    return keywords
