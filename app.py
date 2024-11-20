#!/usr/bin/env python
#coding: utf-8
#!python3

import re
import os

import torch
import gradio as gr
from PIL import Image
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif"]
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Model loaded!")


client = QdrantClient("http://localhost:6333")
collection_name = "image_embeddings"
IMAGES_PER_PAGE = 24  # Number of images to display per page


def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
        return image_features.to(torch.float32).cpu().squeeze(0).numpy()


def search(query):
    with torch.no_grad():
        text_emb = model.get_text_features(
            **processor.tokenizer(query, return_tensors="pt").to(device)
        ).cpu().squeeze(0).numpy()
    
    search_result = client.search(
        collection_name=collection_name,
        query_vector=text_emb.tolist(),
        limit=1000,
        with_payload=True
    )

    image_paths = [hit.payload["image_path"] for hit in search_result]
    display_paths = [os.path.splitext(img_path)[0] for img_path in image_paths]
    return image_paths, display_paths  # Removed page_number_state


def reset_page_number():
    return 1


def update_gallery(image_paths, page_number):
    start_index = (page_number - 1) * IMAGES_PER_PAGE
    end_index = start_index + IMAGES_PER_PAGE
    current_image_paths = image_paths[start_index:end_index]
    gallery_images = [Image.open(img_path) for img_path in current_image_paths]
    return gallery_images


def next_page(image_paths, page_number):
    total_pages = (len(image_paths) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    if page_number < total_pages:
        page_number += 1
    return page_number, update_gallery(image_paths, page_number)


def prev_page(image_paths, page_number):
    if page_number > 1:
        page_number -= 1
    return page_number, update_gallery(image_paths, page_number)


if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="red")) as demo:
        with gr.Row():
            query = gr.Textbox(placeholder="Entrez votre requête (en anglais)")
        with gr.Row():
            search_button = gr.Button("Chercher")
        with gr.Row():
            with gr.Column(scale=3):
                gallery = gr.Gallery(label="Resultats", preview=True)
                with gr.Row():
                    prev_button = gr.Button("Page précédente")
                    next_button = gr.Button("Page suivante")
            with gr.Column(scale=1):
                filename_box = gr.Textbox(label="Nom de l'image sélectionnée", interactive=False)
        image_paths_state = gr.State()
        display_paths_state = gr.State()
        page_number_state = gr.State()

        search_button.click(
            fn=search,
            inputs=query,
            outputs=[image_paths_state, display_paths_state]
        ).then(
            fn=reset_page_number,
            outputs=page_number_state
        ).then(
            fn=update_gallery,
            inputs=[image_paths_state, page_number_state],
            outputs=gallery
        )

        query.submit(
            fn=search,
            inputs=query,
            outputs=[image_paths_state, display_paths_state]
        ).then(
            fn=reset_page_number,
            outputs=page_number_state
        ).then(
            fn=update_gallery,
            inputs=[image_paths_state, page_number_state],
            outputs=gallery
        )

        prev_button.click(
            fn=prev_page,
            inputs=[image_paths_state, page_number_state],
            outputs=[page_number_state, gallery]
        )

        next_button.click(
            fn=next_page,
            inputs=[image_paths_state, page_number_state],
            outputs=[page_number_state, gallery]
        )

    demo.launch(share=True)
