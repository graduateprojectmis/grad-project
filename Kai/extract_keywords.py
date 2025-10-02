import langid
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download("punkt")
nltk.download("stopwords")


def extract_keywords(text) -> list:
    keywords = []
    lang = langid.classify(text)[0]

    if lang == "zh":
        chinese_punctuation = set("，。！？；：\"'（）【】《》〈〉—…、「」")

        dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir, "keywords_dict.txt")
        if os.path.exists(file_path):
            jieba.load_userdict(file_path)

        # Use jieba to extract keywords
        raw_keywords = list(jieba.cut_for_search(text))
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



