# Custom implementation of text summarization and text preprocessing
import re

# Define stopwords and punctuation
STOP_WORDS = set(["a", "about", "above", "after", "again", ...])  # Define your list of stopwords
punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~")

def summarizeText(text):
    """
    Summarize the given text.

    Args:
        text (str): The input text to be summarized.

    Returns:
        dict: A dictionary containing the original raw text, its length, the summarized text, and its length.
    """
    stopwords = STOP_WORDS

    # Tokenize the text into words
    words = text.lower().split()

    # Remove stopwords and punctuation
    words = [word for word in words if word not in stopwords and word not in punctuation]

    # Count word frequency
    word_freq = {}
    for word in words:
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

    max_freq = max(word_freq.values())

    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    # Tokenize the text into sentences
    sentences = text.split(".")

    sent_scores = {}
    for sent in sentences:
        for word in sent.split():
            if word in word_freq:
                if sent not in sent_scores:
                    sent_scores[sent] = word_freq[word]
                else:
                    sent_scores[sent] += word_freq[word]

    select_len = int(len(sentences) * 0.3)

    # Get the top sentences based on scores
    summary = sorted(sent_scores, key=sent_scores.get, reverse=True)[:select_len]

    final_summary = " ".join(summary)
    result = {
        'RawText': text,
        'lenRawText': len(words),
        'summaryText': final_summary,
        'lensummaryText': len(final_summary.split()),
    }

    return result

def textPreProcessing(text, filter1="", filter2="", filter3="", filter4="", filter5="", filter6="", filter7=""):
    """
    Pre-process the input text based on filters.

    Args:
        text (str): The input text to be pre-processed.
        filter1 (str): Filter 1 status ("on" or "off").
        filter2 (str): Filter 2 status ("on" or "off").
        filter3 (str): Filter 3 status ("on" or "off").
        filter4 (str): Filter 4 status ("on" or "off").
        filter5 (str): Filter 5 status ("on" or "off").
        filter6 (str): Filter 6 status ("on" or "off").
        filter7 (str): Filter 7 status ("on" or "off").

    Returns:
        dict: A dictionary containing the summarized text information.
    """
    if filter1:
        text = re.sub(r'\[\d+\]', '', text)

    if filter2:
        text = re.sub(r'\(\d+\)', '', text)

    if filter3:
        text = re.sub(r'\{[a-zA-Z0-9]+\}', '', text)

    if filter4:
        text = re.sub(r'<.*?>', '', text)

    if filter5:
        text = re.sub(r'\*\w+\*', '', text)

    if filter6:
        text = re.sub(r'#\w+', '', text)

    if filter7:
        text = re.sub(r'#\w+', '', text)

    result = summarizeText(text)

    return result