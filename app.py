import speech_to_text as stt
import file_search as fs
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import numpy as np
import recorder as r

app = Flask(__name__)
socketio = SocketIO(app)

class Progress:
    def __init__(self):
        self.progress = "Progress"

    def set_progress(self, progress):
        self.progress = progress

    def get_progress(self):
        return self.progress
_progress = Progress()
recorder = r.Recorder()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/speech_to_text')
def speech_to_text():
    audio_files_list = fs.list_audio_files()
    dataset_list = fs.list_dataset()
    return render_template('speech_to_text.html',
                           audio_files_list=audio_files_list,dataset_list=dataset_list)

# Show progress of speech recognition
@socketio.on('recognition_progress')
def recognition_progress(data):
    #print(data + " is sent to recognition_progress")
    socketio.emit('recognition_progress', {'progress': data})

@app.route('/update_progress', methods=['POST'])
def update_progress():
    return jsonify({'progress': _progress.get_progress()})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    recorder.start_recording()
    return jsonify({"status": "success"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    recorder.stop_recording()
    return jsonify({"status": "success"})

@app.route('/start_recognization_live', methods=['POST'])
def start_recogniztion_live():
    data = request.json
    model = data.get('model')
    prompt = data.get('prompt')
    recognized_text = stt.recognize_speech(filename="temp.wav", model=model, p=prompt, progress=_progress)
    return jsonify({"recognized_text": recognized_text})

@app.route('/start_recognization', methods=['POST'])
def start_recogniztion():
    data = request.json
    audio_file = data.get('file')
    model = data.get('model')
    prompt = data.get('prompt')
    print(audio_file, model, prompt)
    recognized_text = stt.recognize_speech(filename=audio_file, model=model, p=prompt, progress=_progress)
    return jsonify({"recognized_text": recognized_text})

@app.route('/save_to_database', methods=['POST'])
def save_to_database():
    data = request.json
    dataset = data.get('dataset')
    result = data.get('result')
    file = data.get('file')
    filename = file.split(".")[0]
    print(dataset, result)
    # Result text
    with open("database/"+ dataset + '/'+ filename + ".txt", "w") as file:
        file.write(result)
    # Text embedding
    word_embeddings = fs.SentenceTransformer_word_embedding(result)
    with open("database/"+ dataset + '/'+ filename + ".npy", 'wb') as file:
        np.save(file, word_embeddings)

    return jsonify({"status": "success"})


@app.route('/live_speech')
def text_to_speech():
    dataset_list = fs.list_dataset()
    return render_template('live_speech.html',dataset_list=dataset_list)

@app.route('/file_search')
def file_search():
    dataset_list = fs.list_dataset()
    return render_template('file_search.html',dataset_list=dataset_list)

@app.route('/search_by_text_embedding', methods=['POST'])
def search_by_text_embedding():
    data = request.json
    dataset = data.get('dataset')
    input_str = data.get('keyword')
    files = fs.search_by_text_embedding(dataset, input_str)
    return jsonify({"files": files})

@app.route('/search_by_keyword', methods=['POST'])
def search_by_keyword():
    data = request.json
    dataset = data.get('dataset')
    keyword = data.get('keyword')
    files = fs.search_by_keyword(dataset, keyword)
    return jsonify({"files": files})


if __name__ == '__main__':
    socketio.run(app, debug=True)


