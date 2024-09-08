# Multimodal RAG

# Advanced Multimodal ChatBot

## Overview
This project is an Advanced Multimodal ChatBot built using the Crew AI framework, LangChain Groq, Gradio, and various NLP tools. The chatbot can process text queries and PDF files, extract relevant information, perform image searches, and generate responses based on user inputs.

## Features

- **File Upload**: Users can upload PDF files, and the chatbot will analyze the content and answer questions based on the knowledge obtained from the file.
- **Image Search**: The chatbot can perform image searches related to the queries and display the relevant image.
- **Custom Styling**: The Gradio interface has custom CSS for a unique look and feel.
- **Sequential Processing**: Tasks are processed sequentially, ensuring that each step is completed before moving on to the next.

## Requirements
- Python 3.8+
- Gradio
- Crew AI
- LangChain Groq
- Transformers
- SerpAPI
- PIL (Pillow)

## Installation
1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Search Tool
The `search_tool` function extracts the main content from the user's query and uses the SerpAPI to perform a Google search. If the content is related to medical terms, it searches for relevant images and returns the first image found.

### PDF Search Tool
The `create_pdf_search_tool` function creates a tool for searching within a PDF file using a pre-trained model. This tool is used to extract information from uploaded PDF files.

### Gradio Interface
The Gradio interface (`gr.Blocks`) provides a user-friendly way to interact with the chatbot. The UI includes options for uploading files, entering text, and receiving outputs such as text responses and images.

### Running the Application
To run the application, simply execute the following command:
```bash
python app.py
```

### Customization
CSS: The interface's appearance can be customized via the custom_css variable.
JavaScript: The js variable allows for adding custom JavaScript animations to the interface

### Example Workflow
Upload a PDF File: The user uploads a PDF file containing relevant content.
Input a Query: The user enters a query related to the content of the PDF.
Receive Output: The chatbot processes the PDF and returns a formatted response. If relevant, an image related to the query is also displayed.

### Credits
This project utilizes several open-source libraries and tools, including Crew AI, LangChain, Gradio, and SerpAPI. Special thanks to the developers and contributors of these tools.
