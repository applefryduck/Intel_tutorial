import whisper
import tqdm
import numpy as np
import string
import os
import time
from punctuators.models import PunctCapSegModelONNX
from typing import List
from pydub import AudioSegment
import speech_recognition as sr
import app
import recorder as r


def Split_to_chunks(filename):
    # Load audio file
    audio = AudioSegment.from_wav(filename)
    duration = 1
    # Set duration to 5 minutes
    five_minutes = duration * 60 * 1000  # Convert duration to milliseconds
    audio_chunks = []
    print("Splitting the audio")
    app.progress_string = "Splitting the audio"
    # Split every five minutes
    for i in tqdm.trange(0,len(audio),five_minutes):
        audio_chunks.append(audio[i:i+five_minutes])
    # Save the chunks to the specified location
    for i, chunk in enumerate(audio_chunks):
        chunk.export(f"chunks/chunk{i}.wav", format="wav")
    del audio

    return len(audio_chunks)

def remove_punctuation(text):
    # Remove punctuation
    result = text.translate(str.maketrans("", "", string.punctuation))
    return result

def add_punctuation(text):
    # Load the mode from the internet
    m = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    )

    input_texts: List[str] = [text]
    results = m.infer(texts=input_texts, apply_sbd=True)
    return results

def recognize_speech(filename, model, p, progress):
    # load the model
    # The fist time you run this code, it will download the model from the internet
    model = whisper.load_model(model)
    print("Model loaded")
    progress.set_progress("Model loaded")
    results = list()

    t1 = time.time()
    # split the audio into chunks
    print(filename)
    print("Chunks splitting")
    progress.set_progress("Chunks splitting")

    audio_chunks = Split_to_chunks(filename)
    if not os.path.isdir('chunks'):
        os.mkdir('chunks')

    print("Decoding")
    progress.set_progress("Decoding")

    results = ""
    for i in tqdm.tqdm(range(audio_chunks)):
        chunk = whisper.load_audio("chunks/chunk"+str(i)+".wav")
        # decode the audio
        result = model.transcribe(chunk,language="zh",
                                  initial_prompt=p,
                                  condition_on_previous_text=False)
        results += result['text']
        del chunk
    # remove the chunks
    for i in range(audio_chunks):
        os.remove("chunks/chunk"+str(i)+".wav")
    # remove the punctuation of the text
    print("Removing punctuation")
    progress.set_progress("Removing punctuation")
    results = remove_punctuation(results)
    # add the punctuation of the text by model
    print("Adding punctuation")
    progress.set_progress("Adding punctuation")
    results = add_punctuation(results)
    t2 = time.time()
    print("====================================")
    print("Done")
    progress.set_progress("Done")
    print("Model:",model)
    print("Elasped Time:",t2-t1)
    print("Speech to text result:\n")
    result_str = "".join(results[0])
    print(result_str)
    return result_str

def start_record(mic):
    # a function to start the microphone
    r = sr.Recognizer()
    mic = sr.Microphone()


def stop_record():
    # a function to stop the microphone
    r = sr.Recognizer()
    mic = sr.Microphone()

def Live_microphone(model_name,prompt):
    t1 = time.time()
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("Say something...")
        audio = r.listen(source)
        print("Recognizing...")
    text = r.recognize_whisper(audio,model=model_name,prompt=prompt)
    # remove the punctuation of the text
    results = remove_punctuation(text)
    print("Removing punctuation")
    # add the punctuation of the text by model
    results = add_punctuation(results)
    print("Adding punctuation")
    t2 = time.time()
    print("====================================")
    print("Done")
    print("Model:",model_name)
    print("Elasped Time:",t2-t1)
    print("Speech to text result:\n")
    result_str = "".join(results[0])
    print(result_str)