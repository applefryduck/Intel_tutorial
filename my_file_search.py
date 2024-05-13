import os
from sentence_transformers import SentenceTransformer
import re , math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def SentenceTransformer_word_embedding(text):
    # load the word embedding model from the internet
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")    
    words = re.split(",|﹑|，|。|？|！|：|；|、「」:?",text)

    # Get the embeddings for each word
    word_embeddings = model.encode(words)
    return word_embeddings

def make_new_dir(current_dir):
    dir_name = input("Enter the directory name: ")
    os.chdir(current_dir + "/database")
    os.mkdir(dir_name)
    os.chdir("../")
    return

def save_the_file(current_dir,text,dir_list,d):
    # Write the text to the file
    filename = input("Enter the file name: ")
    with open(current_dir + "/database/" + dir_list[int(d)-1]+"/"+filename+".txt", 'w',encoding="utf-8") as f:
        f.write(text)
    # Text embedding
    word_embeddings = SentenceTransformer_word_embedding(text)
    # Write the text embedding to the file
    with open(current_dir + "/database/" + dir_list[int(d)-1]+"/"+filename+".npy", 'wb') as f:
        np.save(f, word_embeddings)
    return

def Save(text):
    print("=============================")
    print("1. Save to file")
    print("2. Don't save")
    print("=============================")
    choice = input("Enter your choice: ")
    if choice == '1':
        # Print directory list
        print("Select a directory: ")
        current_dir = os.getcwd()
        dir_list = os.listdir(current_dir + "/database")
        if(len(dir_list)==0):
            print("No directory found, creating a new directory...")
            # Create a directory if it doesn't exist
            make_new_dir(current_dir=current_dir)
            print("=============================")
            print("Select a directory: ")
        dir_list = os.listdir(current_dir + "/database")
        # Print the directory list
        for i in range(len(dir_list)):
            print(str(i+1)+". "+dir_list[i])
        print(str(len(dir_list)+1)+". "+"Save to a new directory")
        # Select a directory
        d = input("Enter your choice: ")
        if d < str(len(dir_list)+1):
            save_the_file(current_dir,text,dir_list,d)
        elif d == str(len(dir_list)+1):
            make_new_dir(current_dir=current_dir)
            # Write the text to the file
            dir_list = os.listdir(current_dir + "/database")
            save_the_file(current_dir,text,dir_list,d)
    elif choice == '2':
        pass
    else:
        print("Invalid choice")
    return

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

def search_with_file_embedding(dir):
    file_embedding = check_file_embedding(dir)
    while(True):
        # Get the user's input string
        input_str = input("Enter the content of file you are searching for: ")
        # Find the filename with the highest similarity
        most_similar_filename = file_semantic_search(input_str, file_embeddings=file_embedding)
        print("=============================")
        print("The most similar file is: ")
        step = 1
        for(i,j) in most_similar_filename:
            print(str(step) + ". " + i + ".txt")
            step += 1
        print("4. Ask again")
        print("5. Return to menu")
        print("=============================")
        # Which file to open
        c = input("Enter your choice: ")
        if c < str(len(most_similar_filename)+1):
            # Open the file
            os.popen(dir + "/" + most_similar_filename[int(c)-1][0] + ".txt")
            # if the file is opened, return to menu
            return
        elif c == str(4):
            continue
        elif c == str(5):
            return
        else:
            print("Invalid choice")

def search_with_keyword(dir):
    input_str = input("Enter the keyword: ")
    # Find the filename containing the keyword
    file_list = os.listdir(dir)
    valid_files = []
    for file in file_list:
        filename = file.split(".")[0]
        sub = file.split(".")[1]
        if sub == "txt":
            text = open(dir + "/" + filename + ".txt", 'r',encoding="utf-8").read()
            if input_str in text:
                valid_files.append(filename)
    print("=============================")
    print("The files containing the keyword are: ")
    for i in range(len(valid_files)):
        print(str(i+1)+". "+valid_files[i]+".txt")
    print(f"{len(valid_files) + 1}. Return to menu")
    print("=============================")
    # Which file to open
    c = input("Enter your choice: ")
    if c < str(len(valid_files)+1):
        # Open the file
        os.popen(dir + "/" + valid_files[int(c)-1] + ".txt")
        # if the file is opened, return to menu
        return
    elif c == str(len(valid_files)+1):
        return
    else:
        print("Invalid choice")


def file_search():
    print("Select a directory: ")
    current_dir = os.getcwd()
    dir_list = os.listdir(current_dir + "/database")
    if(len(dir_list)==0):
        print("No directory found, return to menu...")
        return
    else:
        # Print the directory list
        for i in range(len(dir_list)):
            print(str(i+1)+". "+dir_list[i])
        # Select a directory
        d = input("Enter your choice: ")
        dir = current_dir + "/database/" + dir_list[int(d)-1]
        print("1. Search by text embedding")
        print("2. Search by keyword")
        sel = input("Enter your choice: ")
        if sel == '1':
            search_with_file_embedding(dir=dir)
        elif sel == '2':
            search_with_keyword(dir=dir)
        else:
            print("Invalid choice")
