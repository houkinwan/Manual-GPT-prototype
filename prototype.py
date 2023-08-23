import numpy as np
import pandas as pd
import pickle
import os
import json
import PyPDF2 as pdf
from PyPDF2 import PdfReader
import re
from tqdm import tqdm
import openai
import time
import warnings
import sys
warnings.filterwarnings('ignore')
from openai.embeddings_utils import get_embedding
from IPython.display import clear_output
from tenacity import AsyncRetrying, RetryError, stop_after_attempt



    

os.system('cls' if os.name == 'nt' else 'clear')
print("Welcome to the Technical Support Chatbot")
openai.api_key = input("Please enter your OpenAI API key:")
os.system('cls' if os.name == 'nt' else 'clear')

MAX_TOKENS = 16385

guide = 'Key in "new" to start a new context session \nKey in "exit" to end the chatbot \nKey in "manuals" to display all available manual \nKey in "load pdfs" to load new pdfs or "load pdfs reset" to reset the database (costly) and load \nKey in "delete" to delete pdfs \nKey in "similarity" to set k-nearest parameters \nKey in "details" to toggle details \nKey in "breadth" to set page search breadth'



# Known Issues
### wrong api key lets you through, but will error when you use the chatbot
### having k nearest values that are too big may cause errors in indexing, also having too big breadth values.
### too big breadth values will easily exceed the token limit, so does having too big k values, so tune accordingly. Default parameters seem good.
### pandas version has to be 1.4.4 or might run into compatibility issues with pickle
### running with an unpaid openai API key will probably result in error because embedding is much slower and will timeout
### general limit testing for the inputs, do not put wrong inputs as errors are not handled properly, this is a WIP
### no error handling so make sure inputs are reasonable, i.e. correct pdf ids, correct ranges of pages, etc
### python needs administrator access for the directory of the script to read and write
### openai might update their policies and the AI models
### it is expensive to chunk large files, so do so sparingly and avoid deleting them
### option within both to amend the prompts could be added

# Priorities
### error handling
### other types of text files for FAQs (.txt)
### reformat how it partitions chunks of text into more uniform and predictable ways
### add more user friendly aspects
def sanity_check():
    #database check
    with open(f"data_json.json", 'r') as openfile:
        data_json = json.load(openfile)
    if not (len(data_json["stored_pdfs"]) == pd.read_pickle("vectors/manual_vectors.csv").shape[0]):
        print("Data incompatibility detected, please do not interact with files directly including deleting folders and csv files. Reset recommended.")
        if input("Reset? (Y/N):").lower() == "y":
            reset_data()
        else:
            print("Exiting")
            exit()
        
        
    #dependency check
    
    
    
def embed(x):
    '''embed with openai'''
    
    try:
        return get_embedding(x, engine="text-embedding-ada-002")
    except RetryError as e:
        os.system('cls' if os.name == 'nt' else 'clear')
        openai.api_key = input("It seems like something is wrong with OpenAI. \nMake sure your internet is connected and please reenter your API key: ")
        os.system('cls' if os.name == 'nt' else 'clear')
        print(guide)
        
        return embed(x)
    


def reset_data():
    '''clear all data from directories and json history'''
    data_json = {"stored_pdfs":{},"stored_ids":{}
    }
    json_object = json.dumps(data_json, indent=4)
    with open(f"data_json.json", "w") as outfile:
        outfile.write(json_object)  
    manual_vectors = pd.DataFrame({"pdf_id":[],"pdf_name":[],"avg_embedding":[]})
    manual_vectors.to_pickle(f'vectors/manual_vectors.csv')
    folder = f"vectors/PageVDs"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            return False
    return True

def load_pdfs(reset=False):
    '''Loads pdfs that have not been loaded into the dataframes. Set reset to True to wipe all data and start again'''
    tqdm.pandas()
    if reset == True:
        reset_data()
    with open(f"data_json.json", 'r') as openfile:
        data_json = json.load(openfile)
    pdf_files = os.listdir(f"pdf_files")
    max_id = len(data_json.keys())
    manual_vectors = pd.read_pickle(f'vectors/manual_vectors.csv')
    count = 0
    for file in pdf_files:
        if file == ".DS_Store":
            continue
        
        if file not in data_json["stored_pdfs"].values():
            count+=1
            if len(data_json["stored_pdfs"].keys()) == 0:
                cur_id = 0
            else:
                cur_id = int(max(data_json["stored_pdfs"].keys()))+1
            data_json["stored_pdfs"][cur_id] = file
            # PYPDF Part
            cur_pdf_text = []
            print(f"processing {file}")
            with open(f"pdf_files/" + file,"rb") as pdf_file:
                reader = PdfReader(pdf_file)
                for j in range(0, len(reader.pages)):
                    cur_string = reader.pages[j].extract_text()
                    cur_pdf_text.append("|start of page {} ".format(j+1)+cur_string + " end of page {}|".format(j+1))
            page_df = pd.DataFrame({"pg_no":list(range(1,len(cur_pdf_text)+1)),"pg_text":cur_pdf_text,"pdf_id":np.full(len(cur_pdf_text),cur_id)})
            print("embedding vectors...")
            max_count = len(page_df["pg_text"])
            def embed_load(x):
                return embed(x)
            page_df["pg_embedding"] = page_df["pg_text"].progress_apply(embed_load)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"{file} complete!")
            manual_vectors = manual_vectors.append({"pdf_id":cur_id,"pdf_name":file,"avg_embedding":np.array(list(page_df["pg_embedding"])).mean(axis=0)},ignore_index=True)
            page_df.to_pickle(f'vectors/PageVDs/{cur_id}.csv')
            
        else:
            print(f"{file} is a duplicate file")
    os.system('cls' if os.name == 'nt' else 'clear')
    manual_vectors.to_pickle(f'vectors/manual_vectors.csv')
    json_object = json.dumps(data_json, indent=4)
    with open(f"data_json.json", "w") as outfile:
        outfile.write(json_object)
        
    if count == 0:
        input("No new files detected, press enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')
        
    elif count == 1:
        input(f"Added {count} new file to the database, press enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        input(f"Added {count} new files to the database, press enter to continue...")
        os.system('cls' if os.name == 'nt' else 'clear')
        
    if count>=1:
        return True
    else:
        return False
        
def delete_file(path):
    try:
        
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (path, e))
        return False
        
def delete_pdf(pdf_id):
    with open(f"data_json.json", 'r') as openfile:
        data_json = json.load(openfile)
    pdf_name = data_json["stored_pdfs"].pop(str(pdf_id))
    
    json_object = json.dumps(data_json, indent=4)
    with open(f"data_json.json", "w") as outfile:
        outfile.write(json_object)
    
    vector_csv_path = f"Vectors/PageVDs/{pdf_id}.csv"
    pdf_path = f"pdf_files/{pdf_name}"
    delete_file(vector_csv_path)
    delete_file(pdf_path)
    manual_df = pd.read_pickle(f'vectors/manual_vectors.csv')
    manual_df = manual_df[manual_df["pdf_id"] != pdf_id]
    manual_df.to_pickle(f'vectors/manual_vectors.csv')
    
def data_empty():
    with open(f"data_json.json", 'r') as openfile:
        data_json = json.load(openfile)
    return len(data_json["stored_pdfs"])==0

    
    
def get_suitable_documents(prompt, k_docs=8, details=False):
    '''Queries into the vector database to find the most suitable documents based on the prompt'''
    embedded_prompt = embed(prompt)
    manual_df = pd.read_pickle(f'vectors/manual_vectors.csv')
    manual_df["cos"] = manual_df["avg_embedding"].apply(lambda x: x@embedded_prompt)
    manual_df = manual_df.sort_values(by="cos",ascending=False)
    if details == True:
        print("looking at:",list(manual_df["pdf_name"])[0:k_docs])
    return list(manual_df["pdf_id"][0:k_docs])
    
    
def get_page_text(pdf_id,pg_no):
    '''Queries into the database to get the text of a pdf given a page'''
    try:
        page_df = pd.read_pickle(f"vectors/PageVDs/{int(pdf_id)}.csv")
        return page_df[page_df["pg_no"] == pg_no].iloc[0]["pg_text"]
    except:
        return ""
    

    
def get_suitable_text(prompt,k_docs=8, k_pages=10, details=False, threshold = 0.7, breadth = 1):
    '''Queries into the vector database to find the most suitable context text based on the prompt'''
    with open(f"data_json.json", 'r') as openfile:
        data_json = json.load(openfile)
    #Find suitable manuals
    embedded_prompt = embed(prompt)
    doc_ids = get_suitable_documents(prompt,k_docs,details)
    relevant_df = pd.DataFrame()
    for doc_id in doc_ids:
        relevant_df = relevant_df.append(pd.read_pickle(f"vectors/PageVDs/{int(doc_id)}.csv"),ignore_index=True)
    #Merge Documents
    relevant_df["pg_cos"] = relevant_df["pg_embedding"].apply(lambda x: np.array(x)@embedded_prompt)
    ranked_filtered_df = relevant_df.sort_values(by="pg_cos",ascending=False)
    idx_pg_pairs = sorted(list(zip(ranked_filtered_df['pdf_id'][0:k_pages],ranked_filtered_df['pg_no'][0:k_pages])))
    idx_pg_dict = {}
    
    for key, value in idx_pg_pairs:
        idx_pg_dict.setdefault(key, []).append(value)
        
    for key,value in idx_pg_dict.items():
        page_list = value
        new_list = []
        #print(page_list)
        for i,page in enumerate(page_list):
            if i == len(page_list)-1:
                new_list.append(page)
                for b in range(int(breadth)):
                    new_list.append(page-(b+1))
                    new_list.append(page+(b+1))
                
                break
            if (page_list[i+1] - page) >= 4:
                new_list.append(page)
                for b in range(int(breadth)):
                    new_list.append(page-(b+1))
                    new_list.append(page+(b+1))
            else:
                for b in range(int(breadth)):
                    new_list.append(page-(b+1))
                for pgno in list(range(page,page_list[i+1])):
                    new_list.append(pgno)
        #print("new",new_list)
        idx_pg_dict[key] = list(set([i for i in new_list if i > 0]))
    if details == True:
        for ID,pages in idx_pg_dict.items():
            print("pages for {}:".format(data_json["stored_pdfs"][str(ID)]),pages)
    final_list = []
    for ID, pages in idx_pg_dict.items():
        for page in pages:
            final_list.append(get_page_text(ID,page))
    return " ".join(final_list)

def chatbot(premise = "You’re a technical support bot that has access to manuals. The manual information is here: ",
            prompt_mod = " Notes: You CAN ONLY USE INFO FROM THE MANUAL TEXT I REPEAT.  \
            You CAN ONLY USE INFO FROM THE MANUAL TEXT. If there is nothing available in \
            the text, SAY I don't have the info. Do not generalize answers. Do not state \
            the page numbers. Remember the answers can be from different manuals. Say \
            'According to my resources' if you understand this along with your response",
            
            k_docs = 8,
            k_pages = 10,
            threshold = 0.7,
            details = False,
            breadth = 1):
    '''Start of the chatbot. Able to modify the premise or prompt'''
    messages=[
    {"role": "system", "content": premise}
      ]
    response_list = []
    load_pdfs()
    print(guide)
    def conversation(response):
        '''The openai api responses'''


        if len(messages) == 1:
            messages[0]["content"] += get_suitable_text(response,k_pages=k_pages,k_docs=k_docs,details=details, breadth=breadth)
        response_list.append(response)
        content = "Question: " + response + prompt_mod
        messages.append({"role": "user", "content": content})
        
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=messages,
            )
        except RetryError as e:
            openai.api_key = input("It seems like somethings wrong with OpenAI. \nReenter your API key: ")
            return False
            
        message = completion["choices"][0]["message"]
        chat_response = completion.choices[0].message.content
        token_usage = completion.usage.total_tokens
        messages.append({"role": "assistant", "content": chat_response})

        return (messages,token_usage)
    
   


    while True:
        
        with open(f"data_json.json", 'r') as openfile:
            data_json = json.load(openfile)
        looper = data_empty()
        while looper:
            looper = data_empty()
            if load_pdfs():
                print(guide)
                break
            if input("No PDFs detected, type exit to quit or press enter to load PDFs: ") == "exit":
                print("Chat Ended")
                return 
        with open(f"data_json.json", 'r') as openfile:
            data_json = json.load(openfile)
            
        
        cur_msg = input("Question: \n")
        if cur_msg == "exit":
            os.system('cls' if os.name == 'nt' else 'clear')
            break
        elif cur_msg == "manuals":
            messages=[{"role": "system", "content": premise}]
            os.system('cls' if os.name == 'nt' else 'clear')
            with open(f"data_json.json", 'r') as openfile:
                data_json = json.load(openfile)
            for stored_pdf in list(data_json["stored_pdfs"].values()):
                print(stored_pdf)
            print("")
            input("Press enter to continue...")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)

            
            
        elif cur_msg == "load pdfs":
            messages=[{"role": "system", "content": premise}]
            load_pdfs()
            print(guide)
        elif cur_msg == "load pdfs reset":
            messages=[{"role": "system", "content": premise}]
            load_pdfs(reset=True)
        elif cur_msg == "new":
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)
            messages=[{"role": "system", "content": premise}]
        elif cur_msg == "delete":
            messages=[{"role": "system", "content": premise}]
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Stored Manuals\n")
            for key,value in data_json["stored_pdfs"].items():
                print(f"{key}: {value}")
            print("")
            print("To delete 1 and 3 and 6, key in: 1,3,6")
            deletes = input("Delete file IDs (press enter to cancel): ")
            if deletes == "":
                os.system('cls' if os.name == 'nt' else 'clear')
                print(guide)
                continue
            while True:
                def check_id_json(x):
                    for key in x:
                        if key not in data_json["stored_pdfs"].keys():
                            return False
                    return True
                if not (''.join(deletes.split(",")).isnumeric()):
                    deletes = input("Character type error, only enter numbers and commas: ")
                elif not check_id_json(deletes.split(",")):
                    deletes = input("PDF id not found, only enter listed IDs: ")
                else:
                    break
            delete_list = [int(i) for i in deletes.split(",")]
            for pdf_id in delete_list:
                delete_pdf(pdf_id)
             
            print(f"{delete_list} deleted")
            input("Press enter to continue...")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)
        elif cur_msg == "similarity":
            messages=[{"role": "system", "content": premise}]
            os.system('cls' if os.name == 'nt' else 'clear')
            print("The similarity is looking a number of most similar documents, and pages of the inputted number.")
            while True:
                try:
                    k_docs = int(input("Nearest documents: "))
                    break
                except:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Please enter an integer")
            while True:
                try:
                    k_pages = int(input("Nearest pages: "))
                    break
                except:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Please enter an integer")
            input(f"Nearest documents set to {k_docs} and nearest pages set to {k_pages}, press enter to continue...")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)
        elif cur_msg == "details":
            messages=[{"role": "system", "content": premise}]
            os.system('cls' if os.name == 'nt' else 'clear')
            details = not details 
            if details:
                print(f"Enabled details")
            else:
                print(f"Disabled details")
            input("Press enter to continue...")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)
        elif cur_msg == "breadth":
            messages=[{"role": "system", "content": premise}]
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Breadth is the number of pages left and right looked at for the recommended pages.")
            while True:
                try:
                    breadth = int(input("Set page breadth: "))
                    break
                except:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Please enter an integer")
            input(f"Breadth set to {breadth}, press enter to continue...")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(guide)
        else:
            cur_conv = conversation(cur_msg)
            if cur_conv == False:
                messages = [{"role": "system", "content": premise}]
                continue
            print("\nChatGPT:\n" + cur_conv[0][-1]["content"])
            print("\n")
            print(f"tokens used: {cur_conv[1]}/{MAX_TOKENS}\n")
            continue
    print("Chat Ended")
       
            
def main():
    
    os.system('cls' if os.name == 'nt' else 'clear')
    sanity_check()
    chatbot(details=False,k_pages=10,k_docs=10,breadth=1)

if __name__ == "__main__":
    main()