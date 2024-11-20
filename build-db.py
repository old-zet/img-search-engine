#!/usr/bin/env python
# coding: utf-8
# !python3

import os
import time
import logging, logging.config

import torch
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from logger import DB_CONSTRUCTION_Logging

ID_COUNTER = 0  # Global id counter


def print_gpu_usage():
    if device == "cuda":
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")


def extract_features_clip(image_paths, batch_size=64):
    all_features = []
    for i in range(0, len(image_paths), batch_size):
        logger.info(f"Processing sub-batch {i // batch_size + 1}")
        batch_paths = image_paths[i: i + batch_size]
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
                    valid_paths.append(path)
                    logger.info(f"Loaded image: {path}")
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                continue

        if not images:
            logger.info("No valid images in this batch, skipping.")
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        features = image_features.cpu().to(torch.float32).numpy()
        all_features.extend(zip(valid_paths, features))
        logger.info(f"Processed sub-batch {i // batch_size + 1}")
    return all_features


def process_images(folders):
    global ID_COUNTER
    start_time = time.time()
    print_gpu_usage()

    all_files = []
    for folder in folders:
        if not os.path.isdir(folder):
            logger.info(f"Folder not found or inaccessible: {folder}")
            continue

        for root, _, files in os.walk(folder):
            all_files.extend(
                [
                    os.path.join(root, f)
                    for f in files
                    if os.path.isfile(os.path.join(root, f))
                ]
            )

    batch_size = 64
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i: i + batch_size]
        batch_embeddings = extract_features_clip(batch_files, batch_size=batch_size)

        if not batch_embeddings:
            logger.info("No embeddings generated for this batch.")
            continue

        points = []
        for _, (path, embedding) in enumerate(batch_embeddings):
            metadata = {"image_path": path}
            point = PointStruct(
                id=ID_COUNTER,
                vector=embedding.tolist(),
                payload=metadata
            )
            points.append(point)
            ID_COUNTER += 1

        logger.info(f"Inserting {len(points)} points into Qdrant...")

        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Inserted batch {i // batch_size + 1}")

    end_time = time.time()
    logger.info(f"Time taken: {(end_time - start_time):.2f} seconds")


def ensure_file_exists(filepath):
    if not os.path.isfile(filepath):
        open(filepath, "a").close()


def find_folders_with_files(parent_folder):
    folders_with_files = []
    for root, dirs, files in os.walk(parent_folder):
        if files:
            folders_with_files.append(root)
    return folders_with_files


if __name__ == "__main__":
    DB_CONSTRUCTION_Logging().start_logging()
    logger = logging.getLogger("DB_CONSTRUCTION_Logging")
    logger.info("CONVERSION SESSION START")

    logger.info("Loading model...")
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=dtype).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    logger.info("Model loaded!")

    logger.info("Managing DB")
    client = QdrantClient(host='localhost', port=6333)
    collection_name = "image_embeddings"
    vector_size = 512

    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{collection_name}' created.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

    # DB_PATH TO DEFINE HERE
    result = find_folders_with_files(DB_PATH)
    for img_folder in result:
        process_images([img_folder])
    logger.info("Log file saved")
    logger.info("CONVERSION SESSION END")

