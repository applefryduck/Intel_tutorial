import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def SentenceTransformer_word_embedding(text):
    # load the word embedding model from the internet
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")    
    words = re.split(",|﹑|，|。|？|！|：|；|、「」:?",text)

    # Get the embeddings for each word
    word_embeddings = model.encode(words)
    return word_embeddings

def list_dataset():
    current_dir = os.getcwd()
    dir_list = os.listdir(current_dir + "/database")
    return dir_list

def list_audio_files():
    current_dir = os.getcwd()
    file_list = []
    for file in os.listdir(current_dir):
        if file.endswith(".wav"):
            file_list.append(file)
    return file_list

def check_file_embedding(path):
    current_dir = path
    # ensures every file is enbedded
    file_embedding = dict()
    for file in os.listdir(current_dir):
        # Get the filename and the subfix
        filename = file.split(".")[0]
        sub = file.split(".")[1]
        if sub == "txt":
            # Check if the file has been embedded
            check_file = os.path.isfile(current_dir + "/" + filename + ".npy")
            # If the file has not been embedded, embed the file
            if check_file == False:
                text = open(current_dir + "/" + filename + ".txt", 'r',encoding="utf-8").read()
                # Text embedding
                word_embeddings = SentenceTransformer_word_embedding(text=text)
                # Write the text embedding to the file
                with open(current_dir + "/" + filename + ".npy", 'wb') as f:
                    np.save(f, word_embeddings)
            # Load the file embedding
            file_embedding[filename] = np.load(current_dir + "/" + filename + ".npy")
            print(filename + " has " + str(len(file_embedding[filename])) + " embeddings")
    return file_embedding

def file_semantic_search(input_string, file_embeddings):
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # Get the embedding for the input string
    input_embedding = model.encode([input_string])
    file_similarity = dict()
    for filename in file_embeddings:
        # Calculate the cosine similarity between the input embedding and every embedding
        similarities = cosine_similarity(input_embedding, file_embeddings[filename])
        # Find the value of the highest similarity for the input
        highest_similarity = np.amax(similarities[0])
        file_similarity[filename] = highest_similarity
    # Find the filenames with the highest top3 similarity
    most_similar_filename = sorted(file_similarity.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(most_similar_filename)
    return most_similar_filename


def search_by_text_embedding(dataset,input_str):
    dir = os.getcwd() + "/database/" + dataset
    file_embedding = check_file_embedding(dir)
    most_similar_filename = file_semantic_search(input_str, file_embeddings=file_embedding)
    file_list = []
    for filename, _ in most_similar_filename:
        file_list.append(filename)
    return file_list

def search_by_keyword(dataset,input_str):
    dir = os.getcwd() + "/database/" + dataset
    file_list = os.listdir(dir)
    valid_files = []
    for file in file_list:
        filename = file.split(".")[0]
        sub = file.split(".")[1]
        if sub == "txt":
            text = open(dir + "/" + filename + ".txt", 'r',encoding="utf-8").read()
            if input_str in text:
                valid_files.append(filename)
    return valid_files