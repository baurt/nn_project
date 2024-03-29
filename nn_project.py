
import streamlit as st
# base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
import torchutils as tu
import json
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from torchvision.models import googlenet, GoogLeNet_Weights
import requests
from io import BytesIO
import torchvision
from PIL import Image
import torch.nn.functional as F


from torchvision.models import googlenet, GoogLeNet_Weights




trnsfrms2 = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor()
    ]
)

st.sidebar.title("Model Selection")
model_options = ["Model 1", "Model 2"]
selected_model = st.sidebar.radio("Select Model", model_options)


if selected_model == "Model 1":
                
                class_names=['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']
                model = model = torchvision.models.resnet18(pretrained=True)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 11)     



                model.load_state_dict(torch.load('weather_classification_resnet18.pth', map_location=torch.device('cpu')))
                st.title("This App recognizes weather based on image provided")
                url = st.text_input("Enter image url")

                
                def get_prediction(path: str) -> str:
                  response = requests.get(path)
                  image = Image.open(BytesIO(response.content)).convert('RGB')
                  input_tensor = trnsfrms2(image)
                  input_batch = input_tensor.unsqueeze(0)
                  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  input_batch = input_batch.to(device)
                  model.eval()
                  with torch.no_grad():
                    output = model(input_batch)
                  probabilities = F.softmax(output[0], dim=0)
                  predicted_class_index = model(input_batch.to(device)).softmax(dim=1).argmax().item()

                  
                  #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
                  st.image(image.resize((400, 400)))
                  st.write(f'Pred class: {class_names[predicted_class_index]}')

                def get_prediction2(uploaded_file: str) -> str:
                  
                  image = Image.open(uploaded_file).convert('RGB')
                  input_tensor = trnsfrms2(image)
                  input_batch = input_tensor.unsqueeze(0)
                  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  input_batch = input_batch.to(device)
                  model.eval()
                  with torch.no_grad():
                    output = model(input_batch)
                  #probabilities = F.softmax(output[0], dim=0)
                  predicted_class_index = model(input_batch.to(device)).softmax(dim=1).argmax().item()

                  
                  #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
                  st.image(image.resize((400, 400)))
                  st.write(f'Pred class: {class_names[predicted_class_index]}')
                st.title("Multiple Image Uploader")
                    
                    # Allow multiple file uploads
                uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
                    
                if uploaded_files is not None:
                        
                        for uploaded_file in uploaded_files:
                            #st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                            get_prediction2(uploaded_file)
                  

                
                if url:  # Check if URL is not empty
                    try:
                        get_prediction(url)
                        
                        # Process the response further if needed
                    except requests.exceptions.MissingSchema:
                        st.warning("Invalid URL format. Make sure to include 'http://' or 'https://'")


elif selected_model == "Model 2":
        model = googlenet(GoogLeNet_Weights)
        model.fc=nn.Linear(1024,1)     



        model.load_state_dict(torch.load('weights.pt', map_location=torch.device('cpu')))
        st.title("This App recognizes 200 bird species")
        url = st.text_input("Enter image url")

        
        def get_prediction(path: str) -> str:
          response = requests.get(path)
          image = Image.open(BytesIO(response.content)).convert('RGB')
          input_tensor = trnsfrms2(image)
          input_batch = input_tensor.unsqueeze(0)
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          input_batch = input_batch.to(device)
          model.eval()
          with torch.no_grad():
            output = model(input_batch)
          probabilities = F.softmax(output[0], dim=0)
          predicted_class_index = round(output.squeeze(-1).sigmoid().item())

          
          #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
          st.image(image.resize((400, 400)))
          if predicted_class_index==0:
            st.write(f'Pred class: {"Cats"}')
          else:
            st.write(f'Pred class: {"Dogs"}')

        def get_prediction2(uploaded_file: str) -> str:
          
          image = Image.open(uploaded_file).convert('RGB')
          input_tensor = trnsfrms2(image)
          input_batch = input_tensor.unsqueeze(0)
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          input_batch = input_batch.to(device)
          model.eval()
          with torch.no_grad():
            output = model(input_batch)
          #probabilities = F.softmax(output[0], dim=0)
          predicted_class_index = round(output.squeeze(-1).sigmoid().item())

          
          #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
          st.image(image.resize((400, 400)))
          if predicted_class_index==0:
            st.write(f'Pred class: {"Cats"}')
          else:
            st.write(f'Pred class: {"Dogs"}')
        st.title("Multiple Image Uploader")
            
            # Allow multiple file uploads
        uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
            
        if uploaded_files is not None:
                
                for uploaded_file in uploaded_files:
                    #st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    get_prediction2(uploaded_file)
          

        
        if url:  # Check if URL is not empty
            try:
                get_prediction(url)
                
                # Process the response further if needed
            except requests.exceptions.MissingSchema:
                st.warning("Invalid URL format. Make sure to include 'http://' or 'https://'")
          





                  



