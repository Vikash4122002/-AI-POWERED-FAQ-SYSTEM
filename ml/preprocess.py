"""
Text Preprocessing Module for FAQ System
Handles cleaning, tokenization, and text normalization
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for NLP tasks
    """

    def __init__(self, language='english', use_lemmatization=True, remove_stopwords=True):
        """
        Initialize the text preprocessor

        Args:
            language: Language for stopwords (default: 'english')
            use_lemmatization: Use lemmatization instead of stemming (default: True)
            remove_stopwords: Remove common stopwords (default: True)
        """
        self.language = language
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords

        # Download required NLTK data
        self._download_nltk_data()

        # Initialize NLTK components
        self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        logger.info(f"TextPreprocessor initialized with: language={language}, "
                    f"lemmatization={use_lemmatization}, remove_stopwords={remove_stopwords}")

    def _download_nltk_data(self):
        """Download required NLTK data if not already present"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            logger.info("NLTK data downloaded successfully")

    def clean_text(self, text):
        """
        Basic text cleaning: lowercase, remove special chars, extra spaces

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def remove_stopwords_from_text(self, tokens):
        """
        Remove stopwords from token list

        Args:
            tokens: List of word tokens

        Returns:
            List of tokens with stopwords removed
        """
        if self.remove_stopwords:
            return [token for token in tokens if token not in self.stop_words]
        return tokens

    def normalize_tokens(self, tokens):
        """
        Apply stemming or lemmatization to tokens

        Args:
            tokens: List of word tokens

        Returns:
            List of normalized tokens
        """
        normalized = []
        for token in tokens:
            if self.use_lemmatization:
                normalized.append(self.lemmatizer.lemmatize(token))
            else:
                normalized.append(self.stemmer.stem(token))
        return normalized

    def tokenize(self, text):
        """
        Tokenize text into words

        Args:
            text: Cleaned text string

        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return text.split()

    def preprocess(self, text, return_string=True):
        """
        Complete preprocessing pipeline

        Args:
            text: Input text
            return_string: If True, return joined string; if False, return tokens

        Returns:
            Preprocessed text as string or list of tokens
        """
        if not text:
            return "" if return_string else []

        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords_from_text(tokens)
        tokens = self.normalize_tokens(tokens)

        if return_string:
            return ' '.join(tokens)
        return tokens

    def preprocess_batch(self, texts, return_string=True):
        """
        Preprocess a batch of texts

        Args:
            texts: List of input texts
            return_string: If True, return joined strings; if False, return tokens

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, return_string) for text in texts]

    def get_preprocessing_summary(self, original_text, processed_text):
        """
        Get a summary of preprocessing changes

        Args:
            original_text: Original text
            processed_text: Processed text

        Returns:
            Dictionary with preprocessing statistics
        """
        original_words = len(original_text.split())
        processed_words = len(processed_text.split())

        return {
            'original_length': len(original_text),
            'processed_length': len(processed_text),
            'original_word_count': original_words,
            'processed_word_count': processed_words,
            'reduction_percentage': (
                (original_words - processed_words) / original_words * 100
            ) if original_words > 0 else 0
        }


def create_sample_preprocessing():
    """Demo function to show preprocessing in action"""

    sample_texts = [
        "How can I reset my password? I forgot it!",
        "Contact support@company.com for help ASAP!!!",
        "What's the refund policy? Need to return item #12345",
        "Check out our website: https://www.company.com/faq",
    ]

    preprocessor1 = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)
    preprocessor2 = TextPreprocessor(use_lemmatization=False, remove_stopwords=False)

    print("=" * 60)
    print("TEXT PREPROCESSING DEMO")
    print("=" * 60)

    for i, text in enumerate(sample_texts, 1):
        print(f"\nOriginal {i}: {text}")

        processed1 = preprocessor1.preprocess(text)
        summary1 = preprocessor1.get_preprocessing_summary(text, processed1)
        print(f"  With lemmatization: {processed1}")
        print(f"    (Reduced by {summary1['reduction_percentage']:.1f}%)")

        processed2 = preprocessor2.preprocess(text)
        summary2 = preprocessor2.get_preprocessing_summary(text, processed2)
        print(f"  With stemming: {processed2}")
        print(f"    (Reduced by {summary2['reduction_percentage']:.1f}%)")


if __name__ == "__main__":
    create_sample_preprocessing()
