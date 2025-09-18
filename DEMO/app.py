from flask import Flask, render_template, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

text_content = ""
classifications = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    global text_content, classifications

    if request.method == 'POST':
        if 'text_file' in request.files:
            text_file = request.files['text_file']
            text_content = text_file.read().decode('utf-8')

        if 'class_file' in request.files:
            class_file = request.files['class_file']
            lines = class_file.read().decode('utf-8').splitlines()
            classifications = {}
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) > 1:
                    class_name = parts[0]
                    literals = parts[1:]
                    classifications[class_name] = literals

    return render_template('index.html', text=text_content, classifications=classifications)

if __name__ == '__main__':
    app.run(debug=True)
