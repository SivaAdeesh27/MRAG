import os
import gradio as gr
from crewai import Agent, Task, Crew, Process
from crewai_tools import EXASearchTool, SerperDevTool, PDFSearchTool
from langchain_groq import ChatGroq
from diffusers import StableDiffusionXLPipeline
import torch
import re
from gradio_client import Client,file,handle_file
from crewai_tools import tool
import os
import gradio as gr
from crewai import Agent, Task, Crew, Process
from crewai_tools import EXASearchTool, SerperDevTool, PDFSearchTool
from langchain_groq import ChatGroq
from gradio_client import Client
from crewai_tools import tool
from collections import defaultdict
import uuid
from PIL import Image
from langchain_community.utilities import GoogleSerperAPIWrapper
from transformers import pipeline
import serpapi
from langchain_core.prompts import ChatPromptTemplate

os.environ['SERPER_API_KEY'] = "dbab3fad5ec37c15ccf2fbe756db009240920ebe"

# Memory management setup
conversation_store = defaultdict(list)
last_k_messages = 4

def add_to_conversation(session_id, message, role="user"):
    conversation_store[session_id].append((role, message))
    if len(conversation_store[session_id]) > last_k_messages:
        conversation_store[session_id] = conversation_store[session_id][-last_k_messages:]

def get_conversation_history(session_id):
    return conversation_store[session_id]

def generate_prompt_with_history(session_id, query):
    history = get_conversation_history(session_id)
    
    if not history:
        return f"User: {query}\n"

    prompt = ""
    # Include only the last interaction pair (question + response)
    if len(history) >= 2:
        last_interaction = history[-2:]
        for role, message in last_interaction:
            prompt += f"{role.capitalize()}: {message}\n"

    prompt += f"User: {query}\n"
    return prompt


# Set up API keys
os.environ['GROQ_API_KEY'] = "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"



# Initialize the language models and tools
llm_text = ChatGroq(model="llama-3.1-70b-versatile", groq_api_key=os.getenv('GROQ_API_KEY'))
# llm_text = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv('GROQ_API_KEY'))

def search_tool(query: str):
    
    """
    Search for the given query
    """
    # search_query = "site:https://courses.lumenlearning.com/  Image of {query} " 
    # params = {

    #     "api_key": "e29437416bc0fc3384843da6dfbf7165b2b30f46448d6f560e124184b63ac0a9", 

    #     "engine": "google", 

    #     "q": search_query,

    # }
    # search_results = serpapi.GoogleSearch(params).get_dict()

    # print("Image:",search_results['inline_images'][0]['original'])
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that takes the query from the user and extract only the main content from the given query which can be useful for image search",
        ),
        ("human", "{query}"),
    ]
    )

    chain = prompt | llm_text
    res = chain.invoke(
        {

            "query": query,
        }
    )
    res = res.content
    result = res.replace('"', '')
    m = llm_text.invoke("only tell whether the term :"+result+" is realted to medical or not.if related say:'yes' else 'no'.")

    if m.content == "yes" or m.content == "Yes":
        search_query = f"site:https://courses.lumenlearning.com/ Image of {result}" 
        params = {

            "api_key": "de985e1e9b76774435d4223eff195f7e65152efd1f8776539b0076c4fe54a364", 

            "engine": "google", 

            "q": search_query,

        }



        search_results = serpapi.GoogleSearch(params).get_dict()
        #print(search_results)
        #print(search_results['inline_images'][0]['original'])
        if 'inline_images' in search_results.keys():
          return (search_results['inline_images'][0]['original'])
        else:
          return None 
    else:
        return None



def create_pdf_search_tool(pdf_path):
    return PDFSearchTool(
        pdf=pdf_path,
        config=dict(
            llm=dict(
                provider="groq",
                config=dict(
                    model="llama-3.1-70b-versatile",
                ),
            ),
            embedder=dict(
                provider="huggingface",
                config=dict(
                    model="BAAI/bge-small-en-v1.5",
                ),
            ),
        )
    )



file_upload_agent = Agent(
    role="FileUploadAgent",
    goal="Analyze uploaded file and provide responses based on the knowleage obtained by uploaded file to the {query}",
    backstory="Expert in extracting and processing information from uploaded file and answering queries.",
    llm=llm_text,
    tools=[],
    allow_delegation=False,
    verbose=True,
    memory=True,
)

file_upload_agent_analyser = Agent(
    role="FileUploadAgentAnalyser",
    goal="Modify the retrieved content to readable and well formatted based on the {query}",
    backstory="Expert in taking the query and retrieved content and generate answers well",
    llm=llm_text,
    allow_delegation=False,
    verbose=True,
    memory=True,
)




file_upload_task = Task(
    description="Analyze the content of the uploaded file and provide insights or answers based on  the {query} from knowledge obtained by the uploaded file",
    expected_output="A detailed response based on the uploaded file content related to the {query} ",
    agent=file_upload_agent
)

file_upload_analyser_task = Task(
    description="Modify the retrieved content from FileUploadAgent with proper formatting and user readable based on the {query}",
    expected_output="retrievent content with proper formatting and readeable based on the {query}",
    agent=file_upload_agent_analyser,
    context = [file_upload_task]
)



crew2 = Crew(
    agents=[file_upload_agent,file_upload_agent_analyser],
    tasks=[file_upload_task,file_upload_analyser_task],
    process=Process.sequential,
    memory=True,
    cache = True,
    embedder={
                "provider": "huggingface",
                "config":{
                        "model": 'BAAI/bge-small-en-v1.5'
                }
        }
)




def route_task(session_id, user_input=None, file_path=None):
    if file_path:
        add_to_conversation(session_id, "User uploaded a PDF file.")
        pdf_search_tool = create_pdf_search_tool(file_path)
        file_upload_agent.tools = [pdf_search_tool]

        prompt = generate_prompt_with_history(session_id, user_input)
        inputs = {'pdf': file_path, 'query': prompt}
        result={}
        result['tasks_output'] = crew2.kickoff(inputs=inputs)
        result['raw'] = search_tool(user_input)
        print(result)
        return result

# Define your custom CSS
custom_css = """
    .gradio-container {
        background-color: #e7c6ff; /* Change the background color */
    }
    button.primary-button {
        background-color: #02c39a !important; /* Change the button color */
        color: white !important; /* Button text color */
        border-radius: 5px !important; /* Rounded corners for the button */
    }
    input[type='text'] {
        background-color: #ffb703 !important; /* Change the input field background color */
        border: 1px solid #fb8500 !important; /* Border color for the input field */
        border-radius: 5px !important; /* Rounded corners for the input field */
    }
    /*textarea {
        background-color: #ffcad4 !important; /* Change the textbox background color */
        border: 1px solid #457b9d !important; /* Border color for the textbox */
        border-radius: 5px !important; /* Rounded corners for the textbox */
    }*/
    .right-column {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .flex-item {
        flex: 1; /* Flex-grow: allow items to grow */
        margin: 10px;
        /*border: 2px solid #023047; Border color */
        /*border-radius: 5px;  Rounded corners */
        background-color: #ffffff; /* Background color */
        padding: 10px; /* Padding inside the box */
    }
"""
js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Welcome to Our Project Demo!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

# Gradio UI Setup (Including session ID for memory management)
with gr.Blocks(css=custom_css,js=js) as demo:
    with gr.Tabs():
        with gr.TabItem("Advanced Multimodal ChatBot"):
            with gr.Row() as app_row:
                with gr.Column(scale=1) as left_column:
                    app_functionality = gr.Dropdown(
                        label="Chatbot functionality",
                        #choices=["Text Query", "File Upload", "Generate Image","Visual Q&A","Audio"],
                        choices=["File Upload"],
                        value="File Upload", interactive=True)
                    input_txt = gr.Textbox(label="Enter message and upload file...", lines=2, show_label=False)
                    file_upload = gr.File(label="Upload PDF file", file_types=['.pdf'], interactive=True)
                    # image_upload =  gr.Image(label="Picture here", type="pil")
            #         audio_input = gr.Audio(
            #     label="Upload or Record Audio",
            #     type="filepath",  # Use 'file' to get the file path
            # )
                    submit_btn = gr.Button(value="Submit")
                    clear_btn = gr.Button(value="Clear")
                    session_id = gr.State()  # Keep track of the session ID for memory

                with gr.Column(scale=8) as right_column:
                    chatbot_output = gr.Markdown(label="Output", elem_classes="flex-item")
                    image_output = gr.Image(label="Related Image", elem_classes="flex-item")  # Image component for displaying images
            def clear_all():
                return "", None, "", None

            clear_btn.click(
                fn=clear_all,
                inputs=[],
                outputs=[input_txt, file_upload,chatbot_output, image_output]
            )


            # Define action on submit
            def handle_submit(input_txt, file_upload, app_functionality, session_id):
                if not session_id:
                    session_id = str(uuid.uuid4())

                result = None
                image_path = None

                if app_functionality == "File Upload":
                    if file_upload:
                       result = route_task(session_id,file_path=file_upload.name, user_input=input_txt)


                # Handle the output
                if isinstance(result, dict) :
                    raw_output = result.get('raw', None)
                    tasks_output = result.get('tasks_output', [])

                    if tasks_output:
                        image_path = raw_output
                        print("Image_path: ",image_path)
                        #image_output.update(value=image_path, visible=True)
                        print("Tasks output",tasks_output.raw)
                        #summary = tasks_output[0].get('summary', 'No summary available.')
                        add_to_conversation(session_id, tasks_output.raw, role="AI")
                        return tasks_output.raw,image_path,session_id
                    else:
                        return "No valid output available.", None, session_id
                else:
                    add_to_conversation(session_id, str(result), role="AI")
                    return str(result), None, session_id


            submit_btn.click(fn=handle_submit, inputs=[input_txt, file_upload, app_functionality, session_id], outputs=[chatbot_output, image_output, session_id])

demo.launch(debug=True)
