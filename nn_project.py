
import streamlit as st
# base

import torch
import torchvision
import torch.nn as nn

from torchvision import transforms as T

import requests
from PIL import Image


import requests
from io import BytesIO
import torchvision
from PIL import Image

import time





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
                model = torchvision.models.resnet18(pretrained=True)
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
                  start_time = time.time()
                  predicted_class_index = model(input_batch.to(device)).softmax(dim=1).argmax().item()
                  end_time = time.time()
                  response_time = round(end_time - start_time,4)

                  
                  #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
                  st.image(image.resize((400, 400)))
                  st.write(f'Pred class: {class_names[predicted_class_index]}')
                  st.write("Response Time:", response_time, "seconds")

                def get_prediction2(uploaded_file: str) -> str:
                  
                  image = Image.open(uploaded_file).convert('RGB')
                  input_tensor = trnsfrms2(image)
                  input_batch = input_tensor.unsqueeze(0)
                  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  input_batch = input_batch.to(device)
                  model.eval()
                  with torch.no_grad():
                    output = model(input_batch)
                  start_time = time.time()
                  predicted_class_index = model(input_batch.to(device)).softmax(dim=1).argmax().item()
                  end_time = time.time()
                  response_time = round(end_time - start_time,4)

                  
                  #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
                  st.image(image.resize((400, 400)))
                  st.write(f'Pred class: {class_names[predicted_class_index]}')
                  st.write("Response Time:", response_time, "seconds")
                  
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
        class_names2=[
    'Black_footed_Albatross',
    'Laysan_Albatross',
    'Sooty_Albatross',
    'Groove_billed_Ani',
    'Crested_Auklet',
    'Least_Auklet',
    'Parakeet_Auklet',
    'Rhinoceros_Auklet',
    'Brewer_Blackbird',
    'Red_winged_Blackbird',
    'Rusty_Blackbird',
    'Yellow_headed_Blackbird',
    'Bobolink',
    'Indigo_Bunting',
    'Lazuli_Bunting',
    'Painted_Bunting',
    'Cardinal',
    'Spotted_Catbird',
    'Gray_Catbird',
    'Yellow_breasted_Chat',
    'Eastern_Towhee',
    'Chuck_will_Widow',
    'Brandt_Cormorant',
    'Red_faced_Cormorant',
    'Pelagic_Cormorant',
    'Bronzed_Cowbird',
    'Shiny_Cowbird',
    'Brown_Creeper',
    'American_Crow',
    'Fish_Crow',
    'Black_billed_Cuckoo',
    'Mangrove_Cuckoo',
    'Yellow_billed_Cuckoo',
    'Gray_crowned_Rosy_Finch',
    'Purple_Finch',
    'Northern_Flicker',
    'Acadian_Flycatcher',
    'Great_Crested_Flycatcher',
    'Least_Flycatcher',
    'Olive_sided_Flycatcher',
    'Scissor_tailed_Flycatcher',
    'Vermilion_Flycatcher',
    'Yellow_bellied_Flycatcher',
    'Frigatebird',
    'Northern_Fulmar',
    'Gadwall',
    'American_Goldfinch',
    'European_Goldfinch',
    'Boat_tailed_Grackle',
    'Eared_Grebe',
    'Horned_Grebe',
    'Pied_billed_Grebe',
    'Western_Grebe',
    'Blue_Grosbeak',
    'Evening_Grosbeak',
    'Pine_Grosbeak',
    'Rose_breasted_Grosbeak',
    'Pigeon_Guillemot',
    'California_Gull',
    'Glaucous_winged_Gull',
    'Heermann_Gull',
    'Herring_Gull',
    'Ivory_Gull',
    'Ring_billed_Gull',
    'Slaty_backed_Gull',
    'Western_Gull',
    'Anna_Hummingbird',
    'Ruby_throated_Hummingbird',
    'Rufous_Hummingbird',
    'Green_Violetear',
    'Long_tailed_Jaeger',
    'Pomarine_Jaeger',
    'Blue_Jay',
    'Florida_Jay',
    'Green_Jay',
    'Dark_eyed_Junco',
    'Tropical_Kingbird',
    'Gray_Kingbird',
    'Belted_Kingfisher',
    'Green_Kingfisher',
    'Pied_Kingfisher',
    'Ringed_Kingfisher',
    'White_breasted_Kingfisher',
    'Red_legged_Kittiwake',
    'Horned_Lark',
    'Pacific_Loon',
    'Mallard',
    'Western_Meadowlark',
    'Hooded_Merganser',
    'Red_breasted_Merganser',
    'Mockingbird',
    'Nighthawk',
    'Clark_Nutcracker',
    'White_breasted_Nuthatch',
    'Baltimore_Oriole',
    'Hooded_Oriole',
    'Orchard_Oriole',
    'Scott_Oriole',
    'Ovenbird',
    'Brown_Pelican',
    'White_Pelican',
    'Western_Wood_Pewee',
    'Sayornis',
    'American_Pipit',
    'Whip_poor_Will',
    'Horned_Puffin',
    'Common_Raven',
    'White_necked_Raven',
    'American_Redstart',
    'Geococcyx',
    'Loggerhead_Shrike',
    'Great_Grey_Shrike',
    'Baird_Sparrow',
    'Black_throated_Sparrow',
    'Brewer_Sparrow',
    'Chipping_Sparrow',
    'Clay_colored_Sparrow',
    'House_Sparrow',
    'Field_Sparrow',
    'Fox_Sparrow',
    'Grasshopper_Sparrow',
    'Harris_Sparrow',
    'Henslow_Sparrow',
    'Le_Conte_Sparrow',
    'Lincoln_Sparrow',
    'Nelson_Sharp_tailed_Sparrow',
    'Savannah_Sparrow',
    'Seaside_Sparrow',
    'Song_Sparrow',
    'Tree_Sparrow',
    'Vesper_Sparrow',
    'White_crowned_Sparrow',
    'White_throated_Sparrow',
    'Cape_Glossy_Starling',
    'Bank_Swallow',
    'Barn_Swallow',
    'Cliff_Swallow',
    'Tree_Swallow',
    'Scarlet_Tanager',
    'Summer_Tanager',
    'Artic_Tern',
    'Black_Tern',
    'Caspian_Tern',
    'Common_Tern',
    'Elegant_Tern',
    'Forsters_Tern',
    'Least_Tern',
    'Green_tailed_Towhee',
    'Brown_Thrasher',
    'Sage_Thrasher',
    'Black_capped_Vireo',
    'Blue_headed_Vireo',
    'Philadelphia_Vireo',
    'Red_eyed_Vireo',
    'Warbling_Vireo',
    'White_eyed_Vireo',
    'Yellow_throated_Vireo',
    'Bay_breasted_Warbler',
    'Black_and_white_Warbler',
    'Black_throated_Blue_Warbler',
    'Blue_winged_Warbler',
    'Canada_Warbler',
    'Cape_May_Warbler',
    'Cerulean_Warbler',
    'Chestnut_sided_Warbler',
    'Golden_winged_Warbler',
    'Hooded_Warbler',
    'Kentucky_Warbler',
    'Magnolia_Warbler',
    'Mourning_Warbler',
    'Myrtle_Warbler',
    'Nashville_Warbler',
    'Orange_crowned_Warbler',
    'Palm_Warbler',
    'Pine_Warbler',
    'Prairie_Warbler',
    'Prothonotary_Warbler',
    'Swainson_Warbler',
    'Tennessee_Warbler',
    'Wilson_Warbler',
    'Worm_eating_Warbler',
    'Yellow_Warbler',
    'Northern_Waterthrush',
    'Louisiana_Waterthrush',
    'Bohemian_Waxwing',
    'Cedar_Waxwing',
    'American_Three_toed_Woodpecker',
    'Pileated_Woodpecker',
    'Red_bellied_Woodpecker',
    'Red_cockaded_Woodpecker',
    'Red_headed_Woodpecker',
    'Downy_Woodpecker',
    'Bewick_Wren',
    'Cactus_Wren',
    'Carolina_Wren',
    'House_Wren',
    'Marsh_Wren',
    'Rock_Wren',
    'Winter_Wren',
    'Common_Yellowthroat']

        model2 = torchvision.models.resnet18(pretrained=True)

        num_ftrs2 = model2.fc.in_features
        model2.fc = nn.Linear(num_ftrs2, 200)
           



        model2.load_state_dict(torch.load('birds_classifications_weights.pt', map_location=torch.device('cpu')))
        st.title("This App recognizes 200 bird species")
        url2 = st.text_input("Enter image url")

        
        def get_prediction21(path: str) -> str:
          response = requests.get(path)
          image = Image.open(BytesIO(response.content)).convert('RGB')
          input_tensor = trnsfrms2(image)
          input_batch = input_tensor.unsqueeze(0)
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          input_batch = input_batch.to(device)
          model2.eval()
          with torch.no_grad():
            output = model2(input_batch)
          start_time = time.time()
          predicted_class_index = model2(input_batch.to(device)).softmax(dim=1).argmax().item()
          end_time = time.time()
          response_time = round(end_time - start_time,4)

          
          #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
          st.image(image.resize((400, 400)))
          st.write(f'Pred class: {class_names2[predicted_class_index]}')
          st.write("Response Time:", response_time, "seconds")

        def get_prediction22(uploaded_file: str) -> str:
          
          image = Image.open(uploaded_file).convert('RGB')
          input_tensor = trnsfrms2(image)
          input_batch = input_tensor.unsqueeze(0)
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          input_batch = input_batch.to(device)
          model2.eval()
          
          start_time = time.time()
          predicted_class_index = model2(input_batch.to(device)).softmax(dim=1).argmax().item()
          end_time = time.time()
          response_time2 = round(end_time - start_time,4)

          
          #img=input_tensor[i].permute(1, 2, 0).numpy() if isinstance(input_tensor, torch.Tensor) else img[i]
          st.image(image.resize((400, 400)))
          st.write(f'Pred class: {class_names2[predicted_class_index]}')
          st.write("Response Time:", response_time2, "seconds")
        st.title("Multiple Image Uploader")
            
            # Allow multiple file uploads
        uploaded_files2 = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
            
        if uploaded_files2 is not None:
                
                for uploaded_file in uploaded_files2:
                    #st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                    get_prediction22(uploaded_file)
          

        
        if url2:  # Check if URL is not empty
            try:
                get_prediction21(url2)
                
                # Process the response further if needed
            except requests.exceptions.MissingSchema:
                st.warning("Invalid URL format. Make sure to include 'http://' or 'https://'")
          





                  



