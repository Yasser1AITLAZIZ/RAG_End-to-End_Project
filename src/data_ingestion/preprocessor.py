import re


class TextPreprocessor:
    """Preprocesses text data."""

    def __init__(self):
        pass

    def process(self, text: str) -> str:
        """
        Clean and normalize text data.

        Args:
            text (str): The raw text to preprocess.

        Returns:
            str: The cleaned and normalized text.
        """
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = self.remove_special_characters(text)
        return text

    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters from text.

        Args:
            text (str): The text to clean.

        Returns:
            str: The text without special characters.
        """
        pattern = r'[^a-zA-Z0-9À-ž\s.,!?\'"-]'
        text = re.sub(pattern, "", text)
        return text
