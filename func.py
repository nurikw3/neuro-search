import os
import yt_dlp

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def extract_frames(video_url, interval_sec=10):
    ydl_opts = {}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(video_url, download=False)
    formats = info_dict.get('formats', None)

    desired_format = next((f for f in formats if f.get('format_note') == '360p'), None)

    if desired_format:
        url = desired_format.get('url', None)
        cap = cv2.VideoCapture(url)

        frame_count_mx = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        time_mx = frame_count_mx * (1/fps) * 1000 

        time_stamp = 0
        
        save_path = 'images'
        frame_count = 0

        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, time_stamp)

            ret, frame = cap.read()

            if time_stamp >= time_mx: break

            if ret:
                frame_count += 1
                cv2.imwrite(os.path.join(save_path, f"frame_{frame_count}_{video_url.split('=')[-1]}&t={round(time_stamp/1000)}s.jpg"), frame)

                time_stamp += interval_sec * 1000

            else: break
        cap.release()
        cv2.destroyAllWindows()


def compute_distance(query_code , image_codes, method='cosine'):
    if method == "cosine":
        similarities = cosine_similarity(query_code.reshape(1,-1), image_codes)
        distances = 1 - similarities
    elif method == "euclidean":
        distances = euclidean_distances(query_code.reshape(1,-1),image_codes)
    
    return distances[0]


def display_top_k_images(list_of_files, distances, k=5):
    top_k_indices = distances.argsort(axis=0)
    image_path_list = []

    j = 0; i = 0

    images_out = []

    while j < k:
        image_path = list_of_files[top_k_indices[i]]
        if i > 0 and abs(distances[top_k_indices[i]] - distances[top_k_indices[i - 1]]) < 0.00025:
            i += 1
            continue
        image_path_list.append(image_path)
        image = Image.open(image_path)
        images_out.append(image)

        j += 1
        i += 1

    return images_out, ['https://www.youtube.com/watch?v=' + image_path.split('_')[-1][:-4] for image_path in image_path_list]