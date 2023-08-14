# Text Summarization and Pre-processing

This project provides Python functions for text summarization and pre-processing. It uses the spaCy library for text analysis and processing. The summarization algorithm extracts important sentences from the input text to create a concise summary. The pre-processing functions allow you to apply various filters to the input text.

## Installation
    Project Folder : https://github.com/AtharvaPawar456/Sem7-Comps-AllPracs/tree/main/NLP%20-%20Natural%20Language%20Processing/text-summarization-project

1. Clone the repository:

   ```bash
   git clone https://github.com/AtharvaPawar456/Sem7-Comps-AllPracs.git
   cd text-summarization

2. Install Dependencies: 
    ```bash
    pip install -r requirements.txt

    # download the spaCy English model
    python -m spacy download en_core_web_sm



## Install the required libraries:
pip install spacy

## Download the spaCy English model:
python -m spacy download en_core_web_sm


## Usage
Summarizing Text
The summarizeText function takes an input text and returns a summary of the text.

## Code:
from summarization import summarizeText

input_text = "Your input text here..."
summary_result = summarizeText(input_text)

print("Original Text:")
print(summary_result['RawText'])
print("Original Text Length:", summary_result['lenRawText'])
print("\nSummary Text:")
print(summary_result['summaryText'])
print("Summary Text Length:", summary_result['lensummaryText'])


## Text Pre-processing
The textPreProcessing function applies filters to the input text based on specified options.


## Code:
from summarization import textPreProcessing

input_text = "Your input text here..."
processed_result = textPreProcessing(input_text, 
        filter1="on", filter2="off", filter3="on",filter4="on", filter5="on", filter6="on", filter7="on")

print("Processed Text:")
print(processed_result['summaryText'])
print("Processed Text Length:", processed_result['lensummaryText'])



# Text Summarization Flask API

This project provides a Flask API for text summarization and pre-processing.

## Installation

    Project Folder : https://github.com/AtharvaPawar456/Sem7-Comps-AllPracs/tree/main/NLP%20-%20Natural%20Language%20Processing/text-summarization-project

1. Clone the repository:

   ```bash
   git clone https://github.com/AtharvaPawar456/Sem7-Comps-AllPracs.git
   cd text-summarization-api

## Install the required libraries:
   pip install Flask

## API Endpoint
    Summarize
    URL: `/summarize`
    Method: POST

    Input JSON:
            {
                "text": "Your input text here...",
                "filters": {
                    "filter1": "on",
                    "filter2": "off",
                    "filter3": "on",
                    "filter4": "off",
                    "filter5": "on",
                    "filter6": "off",
                    "filter7": "on"
                    }
            }

    Output JSON:
            {
                "message": "Success",
                "selected_filters": { ... },
                "text": "Input text here...",
                "len_text": 123,
                "summary_text": "Summarized text...",
                "len_summary_text": 45
            }

## Status Codes:
    200 OK: Successful response
    500 Internal Server Error: If an error occurs during processing

# Usage:
    1. Run the Flask app:
        python app.py
    2. Send a POST request to http://127.0.0.1:5000/summarize with the required JSON input.

    3. The API will return the summarized text and other relevant information.

## Filters

The `textPreProcessing` function supports the following filters:

1. **Square Brackets with Numbers**:
   - Pattern: `r'\[\d+\]'`
   - Example: `[1]`, `[23]`, `[456]`
   - Description: This filter removes text enclosed in square brackets along with any numeric content.

2. **Parentheses with Numbers**:
   - Pattern: `r'\(\d+\)'`
   - Example: `(1)`, `(42)`, `(789)`
   - Description: This filter removes text enclosed in parentheses along with any numeric content.

3. **Curly Braces with Alphanumeric Text**:
   - Pattern: `r'\{[a-zA-Z0-9]+\}'`
   - Example: `{abc123}`, `{text456}`, `{xyz789}`
   - Description: This filter removes text enclosed in curly braces along with any alphanumeric content.

4. **Angle Brackets with Any Characters**:
   - Pattern: `r'<.*?>'`
   - Example: `<tag>`, `<div>`, `<span>`
   - Description: This filter removes HTML-like tags enclosed in angle brackets.

5. **Words Enclosed in Asterisks**:
   - Pattern: `r'\*\w+\*'`
   - Example: `*emphasis*`, `*strong*`, `*highlight*`
   - Description: This filter removes words enclosed in asterisks.

6. **Hash Tags with Alphanumeric Text**:
   - Pattern: `r'#\w+'`
   - Example: `#Python`, `#DataScience`, `#AI`
   - Description: This filter removes hash tags along with any alphanumeric content.

7. **HTML Tags**:
   - Pattern: `r'<[^>]*>'`
   - Example: `<p>`, `<a href="link">`, `<div class="container">`
   - Description: This filter removes HTML tags along with their attributes.

Add descriptions of any additional filters if applicable.

## Contributing
Contributions are welcome! If you have suggestions, improvements, or bug fixes, please create an issue or a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

