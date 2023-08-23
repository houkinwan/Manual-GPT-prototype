# Chatbot for Technical Support and Document Retrieval

This is a Python script that implements a chatbot capable of providing technical support and retrieving information from manuals using OpenAI's GPT-3.5 model. The chatbot can answer questions based on the content of PDF manuals and provide relevant information.

## Features

- Load and process PDF manuals to extract text and generate embeddings.
- Use OpenAI's GPT-3.5 model to provide answers based on user prompts.
- Find the most suitable documents and pages based on the user's input.
- Customize the number of nearest documents and pages to consider.
- Control the level of details in the responses.
- Set the page search breadth to expand the search area.

## Requirements

- Python >3.9
- Libraries: numpy, pandas 1.4.4, PyPDF2, tqdm, openai

## Usage

1. Clone the repository and navigate to the project directory.
2. In an anaconda environment, and run `pip install -r "requirements.txt"`
3. Enter your OpenAI API key (No payment attached may slow performance).
4. Run the script using `python prototype.py`.

## Commands

- `new`: Start a new context session.
- `exit`: End the chatbot.
- `manuals`: Display available PDF manuals.
- `load pdfs`: Load new PDF manuals.
- `load pdfs reset`: Reset the database and load PDF manuals.
- `delete`: Delete PDF manuals from the database.
- `similarity`: Set the number of nearest documents and pages.
- `details`: Toggle details in responses.
- `breadth`: Set page search breadth.

## Note

- The script might require administrative access for reading and writing files.
- Ensure you have the necessary permissions for the script directory.
- Make sure to handle errors and provide correct inputs.
- The script uses OpenAI's GPT-3.5 model, and you need a valid API key.

## Contributors

- [Your Name](https://github.com/houkinwan)

Feel free to contribute and improve the chatbot's functionality!
