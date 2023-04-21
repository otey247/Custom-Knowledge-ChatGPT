from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

os.environ["OPENAI_API_KEY"] = ''

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

messages = []

def chatbot(input_text):
    messages.append({"role": "user", "content": input_text})
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response_text = index.query(input_text, response_mode="compact")
    messages.append({"role": "assistant", "content": response_text.response})

    # Generate an HTML output for the conversation
    conversation = '<div style="display: flex; flex-direction: column;">'
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            conversation += f'<div style="align-self: flex-end; background-color: #007bff; color: white; border-radius: 10px; padding: 5px 10px; margin: 5px;">{content}</div>'
        elif role == "assistant":
            conversation += f'<div style="align-self: flex-start; background-color: #e0e0e0; color: black; border-radius: 10px; padding: 5px 10px; margin: 5px;">{content}</div>'
    conversation += "</div>"
    
    return conversation

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs=gr.outputs.HTML(label="Conversation"),
                     title="Custom-trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)
