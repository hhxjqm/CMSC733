from bert_score import score
import matplotlib.pyplot as plt
import json
import os
from dotenv import load_dotenv
import numpy as np
import torch
import clip
from PIL import Image
import torch.nn as nn

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
load_dotenv()
DATA_SAVE_PATH = os.getenv('DATA_SAVE_PATH')
NUM_PROMPTS = int(os.getenv('NUM_PROMPTS'))
IMAGE_SAVE_PATH = os.getenv('IMAGE_SAVE_PATH')

def np_save(save_path, data):
    with open(DATA_SAVE_PATH+save_path, 'wb') as f:
        np.save(f, data)

def np_load(data1, data2, data3):
    with open(DATA_SAVE_PATH+data1, 'rb') as f1, open(DATA_SAVE_PATH+data2, 'rb') as f2, open(DATA_SAVE_PATH+data3, 'rb') as f3:
        a = np.load(f1)
        b = np.load(f2)
        c = np.load(f3)
    print(f"CWE: {a}, \nRDW: {b}, \nSYN: {c} \n")
        
def plot(data, title, y_label, y_range, save):
    plt.bar(range(len(data)), data)
    plt.title(title)
    plt.xlabel("index of original prompt")
    plt.ylabel(y_label)
    plt.ylim(0, y_range)
    plt.xticks(range(len(data)), [str(i) for i in range(len(data))])
    plt.savefig(f"{save}.png", format='png')
    plt.close()

def plot_combine(data, data2, data3, title, y_label, y_range, save):
    width = 0.2
    gap = 0.05
    ind = range(0, len(data))
    ind = [x + x * gap for x in ind]
    plt.bar([x - width for x in ind], data, width, label='SYN', color='blue')
    plt.bar(ind, data2, width, label='CWE', color='red')
    plt.bar([x + width for x in ind], data3, width, label='RDW', color='green')
    plt.legend()
    plt.title(title)
    plt.xlabel("index of original prompt")
    plt.ylabel(y_label)
    plt.xticks(ind, [f'{i}' for i in range(len(data))])
    plt.ylim(0, y_range)
    plt.savefig(f"{save}.png", format='png')
    plt.close()

def compare_prompt(cands, refs, similarity_save_path, distance_save_path):
    P, R, F1 = score(cands, refs, lang='en', verbose=True)
    np_save(similarity_save_path, F1)
    distance = [1 - F1 for F1 in F1]
    np_save(distance_save_path, distance)
    return distance
    
def compare_image(refs, CWE, RDW, SYN, cos, CWE_list, RDW_list, SYN_list):
    refs_preprocess = preprocess(Image.open(refs)).unsqueeze(0).to(device)
    refs_features = model.encode_image(refs_preprocess)

    CWE_preprocess = preprocess(Image.open(CWE)).unsqueeze(0).to(device)
    CWE_features = model.encode_image(CWE_preprocess)

    RDW_preprocess = preprocess(Image.open(RDW)).unsqueeze(0).to(device)
    RDW_features = model.encode_image(RDW_preprocess)

    SYN_preprocess = preprocess(Image.open(SYN)).unsqueeze(0).to(device)
    SYN_features = model.encode_image(SYN_preprocess)
    
    # Calculate cosine similarity between reference and image features
    similarity_CWE = cos(refs_features[0], CWE_features[0]).item()
    similarity_CWE = (similarity_CWE + 1) / 2

    similarity_RDW = cos(refs_features[0], RDW_features[0]).item()
    similarity_RDW = (similarity_RDW + 1) / 2
    
    similarity_SYN = cos(refs_features[0], SYN_features[0]).item()
    similarity_SYN = (similarity_SYN + 1) / 2
    
    CWE_list.append(similarity_CWE)
    RDW_list.append(similarity_RDW)
    SYN_list.append(similarity_SYN)



def main():
    # Lists to store references and candidate prompts
    refs = []
    cands_CWE = []
    cands_RDW = []
    cands_SYN = []

    # Load data from JSON file
    with open('Results/datas/data.json', 'r') as f:
        json_data = json.load(f)
        
    # Populate lists with data from JSON
    for key, value in json_data.items():
        refs.append(value["p"])
        cands_CWE.append(value["CWE_p"])
        cands_RDW.append(value["RDW_p"])
        cands_SYN.append(value["SYN_p"])

    # Calculate BERTScore for CWE prompts
    distance_F1_CWE = compare_prompt(cands_CWE, refs, "F1_similarity_CWE.npy", "F1_distance_CWE.npy")
    
    # Calculate BERTScore for RDW prompts
    distance_F1_RDW = compare_prompt(cands_RDW, refs, "F1_similarity_RDW.npy", "F1_distance_RDW.npy")

    # Calculate BERTScore for SYN prompts
    distance_F1_SYN = compare_prompt(cands_SYN, refs, "F1_similarity_SYN.npy", "F1_distance_SYN.npy")
        
    # Load F1 scores and distances
    print("\n===================Prompts Distances=====================\n")
    np_load("F1_distance_CWE.npy", "F1_distance_RDW.npy", "F1_distance_SYN.npy")

    # Lists to store image similarities and distances
    image_similarity_CWE = []
    image_similarity_RDW = []
    image_similarity_SYN = []
    cos = torch.nn.CosineSimilarity(dim=0)

    # Calculate image similarities and distances
    for i in range(NUM_PROMPTS):
        folder_name = f"image{i}"
        refs_image = f"{IMAGE_SAVE_PATH}{folder_name}/origin.webp"
        CWE_image = f"{IMAGE_SAVE_PATH}{folder_name}/CWE.webp"
        RDW_image = f"{IMAGE_SAVE_PATH}{folder_name}/RDW.webp"
        SYN_image = f"{IMAGE_SAVE_PATH}{folder_name}/SYN.webp"
        compare_image(refs_image, CWE_image, RDW_image, SYN_image, cos, image_similarity_CWE, image_similarity_RDW, image_similarity_SYN)
        
    distance_image_CWE = [1 - F for F in image_similarity_CWE]
    distance_image_RDW = [1 - F for F in image_similarity_RDW]
    distance_image_SYN = [1 - F for F in image_similarity_SYN]

    # Save results to files
    np_save("image_similarity_CWE.npy", image_similarity_CWE)
    np_save("image_similarity_RDW.npy", image_similarity_RDW)
    np_save("image_similarity_SYN.npy", image_similarity_SYN)
    np_save("image_distance_CWE.npy", distance_image_CWE)
    np_save("image_distance_RDW.npy", distance_image_RDW)
    np_save("image_distance_SYN.npy", distance_image_SYN)
    print("===================Images Distances=====================\n")
    np_load("image_distance_CWE.npy", "image_distance_RDW.npy", "image_distance_SYN.npy")
    
    # Human authorship(P, P’) = abs( delta_(P, P’) / delta_(I, I’) )
    Human_authorship_CWE = [abs(f1 / img) if img != 0 else 0 for f1, img in zip(distance_F1_CWE, distance_image_CWE)]
    np_save("Human_authorship_CWE.npy", Human_authorship_CWE)
    Human_authorship_RDW = [abs(f1 / img) if img != 0 else 0 for f1, img in zip(distance_F1_RDW, distance_image_RDW)]
    np_save("Human_authorship_RDW.npy", Human_authorship_RDW)
    Human_authorship_SYN = [abs(f1 / img) if img != 0 else 0 for f1, img in zip(distance_F1_SYN, distance_image_SYN)]
    np_save("Human_authorship_SYN.npy", Human_authorship_SYN)
    print("===================Human Authorship Score=====================\n")
    np_load("Human_authorship_CWE.npy", "Human_authorship_RDW.npy", "Human_authorship_SYN.npy")
    
    # Visualization
    plot(distance_F1_CWE, "Differences between Original and ContextualWordEmbsAug prompts", "Distance", 0.14, "Results/distant_CWE_prompts")
    plot(distance_F1_RDW, "Differences between Original and RandomWordAug prompts", "Distance", 0.12, "Results/distant_RDW_prompt")
    plot(distance_image_CWE, "Differences between Original and ContextualWordEmbsAug images", "Distance", 0.12, "Results/distant_CWE_images")
    plot(distance_image_RDW, "Differences between Original and RandomWordAug images", "Distance", 0.22, "Results/distant_RDW_images")
    plot(distance_F1_SYN, "Differences between Original and Synonym prompts", "Distance", 0.08, "Results/distant_SYN_prompt")
    plot(distance_image_SYN, "Differences between Original and Synonym images", "Distance", 0.14, "Results/distant_SYN_images")

    plot_combine(distance_F1_SYN, distance_F1_CWE, distance_F1_RDW, "Differences between prompts", "Distance", 0.14, "Results/distant_prompts")
    plot_combine(distance_image_SYN, distance_image_CWE, distance_image_RDW, "Differences between images", "Distance", 0.22, "Results/distant_images")

    plot(Human_authorship_CWE, "Human Authorship scores for ContextualWordEmbsAug", "Score", 2.00, "Results/HA_scores_CWE")
    plot(Human_authorship_RDW, "Human Authorship scores for RandomWordAug", "Score", 3.50, "Results/HA_scores_RDW")
    plot(Human_authorship_SYN, "Human Authorship scores for Synonym", "Score", 2.00, "Results/HA_scores_SYN")
    plot_combine(Human_authorship_SYN, Human_authorship_CWE, Human_authorship_RDW, "Human Authorship scores", "Score", 3.50, "Results/HA_scores")
    
    
    
if __name__ == '__main__':
    main()