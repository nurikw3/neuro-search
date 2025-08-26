import os
import glob
import sys
import shutil

import streamlit as st
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

from func import *


@st.cache_resource
def upload_models():
    from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

    print('The model loading process has started:')
    if torch.cuda.is_available():
        device = torch.device("cuda")  
        print('⚡️ MAKE CUDA GREAT AGAIN!')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   
        print("⚡️ USING FULL POTENTIAL OF APPLE SILICON!")
    else:
        device = torch.device("cpu")
        print('smelled fried..')

    # print(f"Using device: {device}")

    model_id = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_id).to(device) 
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id) 
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, tokenizer, processor, device


def set_page_static_info():
    st.set_page_config(
        page_title="Neuro-Search",
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.title("Neuro Search for YT")


def make_images_and_embedding(video_urls, seconds_step=10):
    if os.path.exists('images'): shutil.rmtree('images')
    os.makedirs('images')

    for url in video_urls:
        extract_frames(url, seconds_step)
    image_folder = 'images'

    image_embeddings = []
    list_of_files = []

    print("Recieve embedding process started:")
    
    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG') :

            image = Image.open(os.path.join(image_folder, filename))
            list_of_files.append(os.path.join(image_folder, filename))

            inputs = processor(text=None, images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features.squeeze(0)
            image_features = image_features.cpu().detach().numpy()

            image_embeddings.append(image_features)
        image_arr = np.vstack(image_embeddings)
        np.save('image_embeddings.npy', image_arr)


def compute_k_nearest_imaget_to_text_prompt(text_imput, top_k):
    prompt = "a photo of " + text_imput
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    text_emb = model.get_text_features(**inputs) 
    query_code = text_emb.squeeze(0).cpu().detach().numpy()
    image_arr = np.load('image_embeddings.npy')

    distances_cosine = compute_distance(query_code, image_arr, method='cosine')

    list_of_files = glob.glob('images/*.jpg')

    images_out, list_of_links = display_top_k_images(list_of_files, distances_cosine, k=top_k)

    return images_out, list_of_links


def compute_k_nearest_imaget_to_image_prompt(image_array, top_k):
    image = processor(text=None, images=image_array, return_tensors="pt")['pixel_values'].to(device) 
    image_features = model.get_image_features(pixel_values=image)

    query_code = image_features.squeeze(0).cpu().detach().numpy()

    image_arr = np.load('image_embeddings.npy')

    distances_cosine = compute_distance(query_code, image_arr, method='cosine')

    list_of_files = glob.glob('images/*.jpg')

    images_out, list_of_links = display_top_k_images(list_of_files, distances_cosine, k=top_k)

    return images_out, list_of_links


def main():
    st.sidebar.title("YT video uploader")
    text_input_url = st.sidebar.text_area("Enter video URLs (each youtube url on a new line):")
    seconds_step = st.sidebar.number_input("Enter the slicing step in seconds:", min_value=1, value=10, step=1)

    if st.sidebar.button("Split video into frames"):
        video_urls = [url.strip() for url in text_input_url.split('\n') if url.strip()]
        with st.spinner('Processing video...'):
            if os.path.exists('image_embeddings.npy'):
                os.remove('image_embeddings.npy')
            print(video_urls)
            make_images_and_embedding(video_urls, seconds_step=seconds_step)
        st.success('Videos from YT uploaded!')

    st.sidebar.markdown('---')

    search_type = st.sidebar.radio("Choose type of search:", ("With text", "With img"))
    
    if search_type == "With text":
        text_input = st.text_input("Enter prompt")
    else:
        image_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    
    top_k = st.number_input("Enter top K", min_value=1, value=5, step=1)

    if st.button(":red[Find a moment from a video by text]"):
        if os.path.exists('images') and os.path.exists('image_embeddings.npy'):
            if search_type == "With text":
                images_out, list_of_links = compute_k_nearest_imaget_to_text_prompt(text_input, top_k)
                for image, link in zip(images_out, list_of_links):
                    col_first, col_second, _  = st.columns(3)
                    with col_first:
                        st.image(image)
                    with col_second:
                        st.markdown(f'<a href="{link}" target="_blank">{link}</a>', unsafe_allow_html=True)
            else:
                if image_file is not None:
                    image = Image.open(image_file)
                    image_array = np.array(image.convert('RGB'))
                    images_out, list_of_links = compute_k_nearest_imaget_to_image_prompt(image_array, top_k)
                    for image, link in zip(images_out, list_of_links):
                        col_first, col_second, _  = st.columns(3)
                        with col_first:
                            st.image(image)
                        with col_second:
                            st.markdown(f'<a href="{link}" target="_blank">{link}</a>', unsafe_allow_html=True)
                else: 
                    st.error('Failed to load image')
        else:
            st.error('Please enter the button. Frames not aviable')



if __name__ == "__main__":
    set_page_static_info()
    model, tokenizer, processor, device = upload_models()
    main()