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

# Disable warnings
warnings.filterwarnings('ignore')

# Load OpenAI API key
openai.api_key = input("Enter API key:")

# Clear terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Constants
MAX_TOKENS = 16385

# Load PDFs and embeddings
def load_pdfs(reset=False):
    if reset:
        reset_data()

    # Load existing data
    data_json = load_json('data_json.json')
    manual_vectors = pd.read_pickle('vectors/manual_vectors.csv')

    pdf_files = [file for file in os.listdir("pdf_files") if file != ".DS_Store"]
    
    for file in pdf_files:
        if file in data_json["stored_pdfs"].values():
            print(f"{file} is a duplicate file")
            continue

        cur_id = max(data_json["stored_pdfs"].keys(), default=-1) + 1
        data_json["stored_pdfs"][cur_id] = file
        
        cur_pdf_text = extract_text_from_pdf(file)
        page_df = create_page_df(cur_pdf_text, cur_id)
        embed_pages(page_df)
        
        manual_vectors = update_manual_vectors(manual_vectors, cur_id, file, page_df)

    save_data_json(data_json)
    save_manual_vectors(manual_vectors)

# Extract text from PDF
def extract_text_from_pdf(file):
    cur_pdf_text = []
    with open(f"pdf_files/" + file, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        for j in range(len(reader.pages)):
            cur_string = reader.pages[j].extract_text()
            cur_pdf_text.append("|start of page {} ".format(j + 1) + cur_string + " end of page {}|".format(j + 1))
    return cur_pdf_text

# Create DataFrame for pages
def create_page_df(cur_pdf_text, cur_id):
    page_df = pd.DataFrame({
        "pg_no": list(range(1, len(cur_pdf_text) + 1)),
        "pg_text": cur_pdf_text,
        "pdf_id": np.full(len(cur_pdf_text), cur_id)
    })
    return page_df

# Embed pages using OpenAI
def embed_pages(page_df):
    def embed_load(x):
        return embed(x)

    page_df["pg_embedding"] = page_df["pg_text"].progress_apply(embed_load)

# Update manual vectors
def update_manual_vectors(manual_vectors, cur_id, file, page_df):
    avg_embedding = np.array(list(page_df["pg_embedding"])).mean(axis=0)
    manual_vectors = manual_vectors.append({"pdf_id": cur_id, "pdf_name": file, "avg_embedding": avg_embedding}, ignore_index=True)
    return manual_vectors

# Save and load JSON
def save_data_json(data_json):
    with open('data_json.json', 'w') as outfile:
        json.dump(data_json, outfile, indent=4)

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as openfile:
            return json.load(openfile)
    return {"stored_pdfs": {}, "stored_ids": {}}

# Clear all data
def reset_data():
    data_json = {"stored_pdfs": {}, "stored_ids": {}}
    save_data_json(data_json)

    manual_vectors = pd.DataFrame({"pdf_id": [], "pdf_name": [], "avg_embedding": []})
    save_manual_vectors(manual_vectors)

    clear_directory('vectors/PageVDs')

# Clear a directory
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        delete_file(file_path)

# Delete a file or directory
def delete_file(path):
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f'Failed to delete {path}. Reason: {e}')

# Embed with OpenAI
def embed(x):
    return get_embedding(x, engine="text-embedding-ada-002")

# Display suitable documents
def get_suitable_documents(prompt, k_docs=8, details=False):
    embedded_prompt = embed(prompt)
    manual_df = pd.read_pickle('vectors/manual_vectors.csv')
    manual_df["cos"] = manual_df["avg_embedding"].apply(lambda x: x @ embedded_prompt)
    manual_df = manual_df.sort_values(by="cos", ascending=False)
    
    if details:
        print("Looking at:", list(manual_df["pdf_name"][:k_docs]))
    
    return list(manual_df["pdf_id"][:k_docs])

# Display suitable text
def get_suitable_text(prompt, k_docs=8, k_pages=10, details=False, threshold=0.7, breadth=1):
    data_json = load_json('data_json.json')
    
    embedded_prompt = embed(prompt)
    doc_ids = get_suitable_documents(prompt, k_docs, details)
    relevant_df = create_relevant_df(doc_ids)
    
    ranked_filtered_df = rank_pages(relevant_df, embedded_prompt)
    idx_pg_dict = create_index_page_dict(ranked_filtered_df, data_json, k_pages, breadth)
    
    final_list = get_final_text(idx_pg_dict, data_json)
    return " ".join(final_list)

# Create DataFrame for relevant pages
def create_relevant_df(doc_ids):
    relevant_df = pd.DataFrame()
    for doc_id in doc_ids:
        relevant_df = relevant_df.append(pd.read_pickle(f"vectors/PageVDs/{int(doc_id)}.csv"), ignore_index=True)
    return relevant_df

# Rank pages by cosine similarity
def rank_pages(relevant_df, embedded_prompt):
    relevant_df["pg_cos"] = relevant_df["pg_embedding"].apply(lambda x: np.array(x) @ embedded_prompt)
    return relevant_df.sort_values(by="pg_cos", ascending=False)

# Create dictionary with index and page number pairs
def create_index_page_dict(ranked_filtered_df, data_json, k_pages, breadth):
    idx_pg_pairs = sorted(list(zip(ranked_filtered_df['pdf_id'][:k_pages], ranked_filtered_df['pg_no'][:k_pages])))
    idx_pg_dict = {}
    
    for key, value in idx_pg_pairs:
        idx_pg_dict.setdefault(key, []).append(value)
        
    for key, value in idx_pg_dict.items():
        page_list = value
        new_list = []
        for i, page in enumerate(page_list):
            if i == len(page_list) - 1:
                new_list.append(page)
                for b in range(int(breadth)):
                    new_list.append(page - (b + 1))
                    new_list.append(page + (b + 1))
                break
            
            if (page_list[i + 1] - page) >= 4:
                new_list.append(page)
                for b in range(int(breadth)):
                    new_list.append(page - (b + 1))
                    new_list.append(page + (b + 1))
            else:
                for b in range(int(breadth)):
                    new_list.append(page - (b + 1))
                
                for pgno in list(range(page, page_list[i + 1])):
                    new_list.append(pgno)
        
        idx_pg_dict[key] = list(set([i for i in new_list if i > 0]))
    
    if details:
        for ID, pages in idx_pg_dict.items():
            print("Pages for {}:".format(data_json["stored_pdfs"][str(ID)]), pages)
    
    return idx_pg_dict

# Get final text for responses
def get_final_text(idx_pg_dict, data_json):
    final_list = []
    for ID, pages in idx_pg_dict.items():
        for page in pages:
            final_list.append(get_page_text(ID, page))
    return final_list

# Get page text
def get_page_text(pdf_id, pg_no):
    try:
        page_df = pd.read_pickle(f"vectors/PageVDs/{int(pdf_id)}.csv")
        return page_df[page_df["pg_no"] == pg_no].iloc[0]["pg_text"]
    except:
        return ""

# Chatbot interaction
def chatbot():
    print("Welcome to the Technical Support Chatbot")
    print("Enter 'exit' to end the chat")

    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Chat ended.")
            break
        
        response = generate_response(user_input)
        print("Chatbot:", response)

# Generate chatbot response
def generate_response(user_input):
    response = ""
    
    if len(user_input) == 0:
        response = "Please enter a valid question."
    elif user_input.lower() == 'load pdfs':
        load_pdfs()
        response = "PDFs loaded successfully."
    elif user_input.lower() == 'help':
        response = "You can enter your questions or type 'exit' to end the chat."
    else:
        response = get_suitable_text(user_input)
    
    return response

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    chatbot()

if __name__ == "__main__":
    main()