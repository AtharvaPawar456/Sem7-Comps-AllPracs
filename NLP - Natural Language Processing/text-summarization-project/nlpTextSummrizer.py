import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import re

def summarizeText(text):
    """
    Summarize the given text.

    Args:
        text (str): The input text to be summarized.

    Returns:
        dict: A dictionary containing the original raw text, its length, the summarized text, and its length.
    """
    # text = textPreProcessing(text)
    stopwords = list(STOP_WORDS)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_freq = max(word_freq.values())

    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq

    sen_tokens = [sent for sent in doc.sents]

    sent_scores = {}
    for sent in sen_tokens:
        for word in sent:
            if word.text in  word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    select_len = int(len(sen_tokens) * 0.3)

    summary = nlargest(select_len, sent_scores, key=sent_scores.get)

    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    result = {
        'RawText': text,
        'lenRawText': len(text.split(' ')),
        'summaryText': summary,
        'lensummaryText': len(summary.split(' ')),
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
