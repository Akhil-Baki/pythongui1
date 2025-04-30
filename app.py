import os
from flask import Flask, render_template, request, send_file
import fitz  # PyMuPDF
from gtts import gTTS
import pyttsx3
from docx import Document
import uuid
from transformers import pipeline  # For text summarization

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = os.path.abspath('outputs')
SUMMARIZE_FOLDER = os.path.abspath('summarize')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SUMMARIZE_FOLDER, exist_ok=True)

# Initialize the summarization pipeline
summarizer = pipeline("summarization")


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        text = ''.join([page.get_text() for page in doc])
        doc.close()
        return text.strip()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    return ""


def convert_text_to_speech(text, mode='online', lang='en'):
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    if mode == 'offline':
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    else:
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)

    return output_path


def summarize_text(text, max_length=130, min_length=30):
    """Summarize the given text using a pre-trained model."""
    try:
        # Ensure the text is not too short or too long
        if len(text.split()) < min_length:
            return "Error: Text is too short to summarize."

        # Split text into manageable chunks if it's too long
        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        summaries = []

        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])

        return ' '.join(summaries)
    except Exception as e:
        return f"Error summarizing text: {e}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        tts_mode = request.form.get('mode', 'online')
        lang = request.form.get('lang', 'en')

        if not file:
            return render_template('index.html', error="No file uploaded")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.pdf', '.docx']:
            return render_template('index.html', error="Invalid file type")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        text = extract_text(file_path)
        if not text:
            return render_template('index.html', error="No readable text found")

        output_path = convert_text_to_speech(text, mode=tts_mode, lang=lang)
        return render_template('index.html', download=output_path)

    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    file = request.files.get('file')

    if not file:
        return render_template('index.html', error="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext != '.pdf':
        return render_template('index.html', error="Only PDF files are supported for summarization")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        return render_template('index.html', error=f"Failed to save file: {e}")

    if not os.path.exists(file_path):
        return render_template('index.html', error="Uploaded file could not be found")

    text = extract_text(file_path)
    if not text:
        return render_template('index.html', error="No readable text found in the PDF")

    summary = summarize_text(text)

    # Save the summary to a file in the summarize folder
    summary_filename = f"summary_{uuid.uuid4().hex}.txt"
    summary_path = os.path.join(SUMMARIZE_FOLDER, summary_filename)
    try:
        with open(summary_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write(summary)
    except Exception as e:
        return render_template('index.html', error=f"Failed to save summary: {e}")

    return render_template('index.html', summary=summary, summary_file=summary_filename, success="PDF successfully summarized and saved.")

if __name__ == '__main__':
    
    port = int(os.environ.get('PORT', 10000))  
    app.run(host='0.0.0.0', port=port)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)


@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """Serve audio files directly from the outputs folder."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return f"Error: The file '{filename}' does not exist.", 404
    return send_file(file_path, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
