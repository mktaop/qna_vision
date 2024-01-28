#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:37:42 2024

@author: avi_patel
"""

import streamlit as st
import os, fitz, io, PyPDF2, uuid, base64, webbrowser, PIL.Image
from openai import OpenAI
import google.generativeai as genai
import google.ai.generativelanguage as glm
import open_clip
from PIL import Image 
import chromadb
import numpy as np
from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from io import BytesIO
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough,RunnableParallel
path = "Path to your PDF document"
path2 = 'Path to where you want to store your JPEGS, good idea to create a folder in the above path. So path and path2 are not same.'
os.chdir(path)


def setup():
    page_title="Chat with charts and tables in your document"
    page_icon=":robot_face:"
    layout="wide"
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
    st.title(page_icon + "  " + page_title + "  " + page_icon)
    st.sidebar.title("Options")


def select_llm():
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-4-vision-preview",
                                   "gemini-pro-vision"),)
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    tokens = st.sidebar.slider("Tokens returned:", min_value=50,
                                    max_value=1000, value=300, step=50)
    return model_name, temperature, tokens


def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
   except OSError:
     print("Error occurred while deleting files.")


def setup_documents(pdf_file_path):
    to_delete_path = path2
    delete_files_in_directory(to_delete_path)
    doc = fitz.open(pdf_file_path)
    os.chdir(to_delete_path)
    for page in doc: 
        pix = page.get_pixmap(matrix=fitz.Identity, dpi=None, 
                              colorspace=fitz.csRGB, clip=None, alpha=False, annots=True) 
        pix.save("pdfimage-%i.jpg" % page.number) 


def get_method():
    method = st.sidebar.radio("Choose method:",
                              ("No Embeddings",
                               "Embeddings"))
    return method
   
 
def base64_vision(prompt, model, temperature, tokens):
    response = client2.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=tokens,
    )
    st.markdown(response.choices[0].message.content)
  

def convert_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


def split_image_text_types(docs):
    
    def is_base64(s):
        try:
            return base64.b64encode(base64.b64decode(s)) == s.encode()
        except Exception:
            return False
    
    def resize_base64_image(base64_string, size=(128, 128)):
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        resized_img = img.resize(size, Image.LANCZOS)
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  
        if is_base64(doc):
            images.append(
                resize_base64_image(doc, size=(480, 480)) 
            )  
        else:
            text.append(doc)
    return {"images": images, "texts": text}


def prompt_func(data_dict):
    messages = []
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)
    return [HumanMessage(content=messages)]

  
def main():
    """ TO DO:
        1. set up streamlit sandbox
        2. 5 tests: 1-use gpt4v on one, 2-use gemini on one, 3-use clipembeddings on all, 4-use gpt4v on all, 5-use gemini on all
        3. sidebar options for different models and parameter choices 
        4. Bonus: use image to generate code

        Returns
        -------
        None.
        """
    
    setup()
    model, temperature, tokens = select_llm()
    uploaded_file = st.file_uploader("Upload your PDF document")
    if uploaded_file:
        setup_documents(uploaded_file.name)
        method = get_method()
        question = st.text_input("Enter your prompt/question and hit return","")
        if question:
            if method=="No Embeddings":
                if model=="gpt-4-vision-preview":
                    base64frames = []
                    for filename in sorted(os.listdir(path2)):
                        file_path = os.path.join(path2, filename)
                        if os.path.isfile(file_path):
                            encoded_image = convert_to_base64(file_path)
                            base64frames.append(encoded_image)
                    prompt = [
                        {
                            "role": "user",
                            "content": [
                                f"{question}",     
                                *map(lambda x: {"image": x, "resize": 480}, base64frames),
                            ],
                        },
                    ]
                    base64_vision(prompt, model, temperature, tokens)
                else:
                    base64frames = []
                    for filename in sorted(os.listdir(path2)):
                        file_path = os.path.join(path2, filename)
                        if os.path.isfile(file_path):
                            encoded_image = PIL.Image.open(file_path)
                            base64frames.append(encoded_image)
                    base64frames.insert(0, question)
                    client = genai.GenerativeModel(model_name=model)
                    responses = client.generate_content(base64frames,generation_config=genai.types.GenerationConfig
                                                        (temperature=temperature, max_output_tokens=tokens))
                    responses.resolve()
                    st.markdown(responses.text)
                    
            else:
                vectorstore = Chroma(
                    collection_name="mm_rag_clip_photos", embedding_function=OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
                )
                image_uris = sorted(
                    [
                        os.path.join(path2, image_name)
                        for image_name in os.listdir(path2)
                        if image_name.endswith(".jpg")
                    ]
                )
                vectorstore.add_images(uris=image_uris)
                retriever = vectorstore.as_retriever()
                model = ChatOpenAI(temperature=temperature,
                                   openai_api_key=OPENAI_API_KEY,
                                   model="gpt-4-vision-preview",
                                   max_tokens=300)
                chain = (
                    {
                        "context": retriever | RunnableLambda(split_image_text_types),
                        "question": RunnablePassthrough(),
                    }
                    | RunnableParallel({"response":prompt_func| model| StrOutputParser(),
                                      "context": itemgetter("context"),})
                )
                response = chain.invoke(question)
                st.markdown(response['response'])
            
            
if __name__ == "__main__":
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client2 = OpenAI(api_key=OPENAI_API_KEY)
    main()