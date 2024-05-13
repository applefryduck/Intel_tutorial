import my_text_to_speech as mts
import my_file_search as mfs
import torch
import os

def print_menu():
    print("  __  __                        ")
    print(" |  \/  |                       ")
    print(" | \  / |   ___   _ __    _   _ ")
    print(" | |\/| |  / _ \ | '_ \  | | | |")
    print(" | |  | | |  __/ | | | | | |_| | ")
    print(" |_|  |_|  \___| |_| |_|  \__,_| \n")
    print("Type 1 ~ 4 to select the function")
    print("=============================")
    print("1. speech to text")
    print("2. live-microphone to text")
    print("3. file search")
    print("4. exit")
    print("=============================")
    choice = input("Enter your choice: ")
    return choice

def list_audio_files():
    current_dir = os.getcwd()
    file_list = []
    for file in os.listdir(current_dir):
        if file.endswith(".wav"):
            file_list.append(file)
    return file_list
def print_stt(model_list):
    print("=============================")
    # list all available files
    file_list = list_audio_files()
    print("Select the file: ")
    for i in range(len(file_list)):
        print(str(i+1)+". "+file_list[i])

    c = input("Select the file to be recognized: ")
    f = file_list[int(c)-1]
    print("Select the model: ")
    for i in range(len(model_list)):
        print(str(i+1)+". "+model_list[i])
    m = input("Enter your choice: ")
    prompt = input("Enter the prompt: ")
    print("=============================")
    return f,m,prompt

def print_live_microphone(model_list):
    print("=============================")
    print("Select the model: ")
    for i in range(len(model_list)):
        print(str(i+1)+". "+model_list[i])
    m = input("Enter your choice: ")
    prompt = input("Enter the prompt: ")
    print("=============================")
    return m,prompt

def main():
    print("CUDA enabled:", torch.cuda.is_available())

    while(True):
        choice = print_menu()
        model_list = ["base","small","medium"]

        if choice == '1':
            f,m,prompt = print_stt(model_list = model_list)
            # convert speech to text
            text = mts.SpeechToText(filename = f,model_name = model_list[int(m)-1],p = prompt)
            # save text file to the specified location
            mfs.Save(text)
        elif choice == '2':
            m,prompt = print_live_microphone(model_list = model_list)
            # convert speech to text
            text = mts.Live_microphone(model_name=model_list[int(m)-1],prompt=prompt)
            # save text file to the specified location
            mfs.Save(text)
        elif choice == '3':
            mfs.file_search()
        elif choice == '4':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()