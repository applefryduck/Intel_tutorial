import speech_to_text as stt
import file_search as fs
import numpy as np
#recognized_text = stt.recognize_speech(filename="input1.wav", model="base", p="")
files = fs.search_by_text_embedding("tale2", "王子")

for file in files:
    print(file)
    print("=====================================")
files = fs.search_by_keyword("tale2", "王子")
for file in files:
    print(file)
    print("=====================================")