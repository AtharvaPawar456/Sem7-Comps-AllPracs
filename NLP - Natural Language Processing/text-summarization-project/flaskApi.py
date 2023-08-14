from flask import Flask, request, jsonify
from nlpTextSummrizer import textPreProcessing

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()

        text = data.get('text', '')
        filters = data.get('filters', {})

        result = textPreProcessing(
            text,
            filters.get('filter1', ""),
            filters.get('filter2', ""),
            filters.get('filter3', ""),
            filters.get('filter4', ""),
            filters.get('filter5', ""),
            filters.get('filter6', ""),
            filters.get('filter7', "")
        )

        response_data = {
            'message': 'Success',
            'selected_filters': filters,
            'text': text,
            'len_text': len(text),
            'summary_text': result["summaryText"],
            'len_summary_text': result["lensummaryText"],
        }

        return jsonify(response_data), 200

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
