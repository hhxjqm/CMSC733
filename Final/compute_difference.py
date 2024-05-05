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

# Lists to store references and candidate prompts
refs = []
cands_CWE = []
cands_RDW = []

# Load data from JSON file
with open('Results/data.json', 'r') as f:
    json_data = json.load(f)
    
# Populate lists with data from JSON
for key, value in json_data.items():
    refs.append(value["p"])
    cands_CWE.append(value["CWE_p"])
    cands_RDW.append(value["RDW_p"])

# Calculate BERTScore for CWE prompts
P_CWE, R_CWE, F1_CWE = score(cands_CWE, refs, lang='en', verbose=True)
with open('Results/F1_similarity_CWE.npy', 'wb') as f:
    np.save(f, F1_CWE)
distance_F1_CWE = [1 - F1 for F1 in F1_CWE]
with open('Results/F1_distance_CWE.npy', 'wb') as f:
    np.save(f, distance_F1_CWE)
    
# Calculate BERTScore for RDW prompts
P_RDW, R_RDW, F1_RDW = score(cands_RDW, refs, lang='en', verbose=True)
with open('Results/F1_similarity_RDW.npy', 'wb') as f:
    np.save(f, F1_RDW)
distance_F1_RDW = [1 - F1 for F1 in F1_RDW]
with open('Results/F1_distance_RDW.npy', 'wb') as f:
    np.save(f, distance_F1_RDW)
    
# Load F1 scores and distances
with open('Results//F1_distance_CWE.npy', 'rb') as f1, open('Results/F1_distance_RDW.npy', 'rb') as f2:
    a = np.load(f1)
    b = np.load(f2)
print(f"CWE: {a}, \nRDW: {b}")

# Lists to store image similarities and distances
image_similarity_CWE = []
image_similarity_RDW = []
num_prompts = int(os.getenv('NUM_PROMPTS'))
cos = torch.nn.CosineSimilarity(dim=0)

# Calculate image similarities and distances
for i in range(num_prompts):
    folder_name = f"image{i}"
    refs_image = f"new_image/{folder_name}/origin.webp"
    CWE_image = f"new_image/{folder_name}/CWE.webp"
    RDW_image = f"new_image/{folder_name}/RDW.webp"
    
    refs_preprocess = preprocess(Image.open(refs_image)).unsqueeze(0).to(device)
    refs_features = model.encode_image(refs_preprocess)

    CWE_preprocess = preprocess(Image.open(CWE_image)).unsqueeze(0).to(device)
    CWE_features = model.encode_image(CWE_preprocess)

    RDW_preprocess = preprocess(Image.open(RDW_image)).unsqueeze(0).to(device)
    RDW_features = model.encode_image(RDW_preprocess)
    
    # Calculate cosine similarity between reference and image features
    similarity_CWE = cos(refs_features[0], CWE_features[0]).item()
    similarity_CWE = (similarity_CWE + 1) / 2

    similarity_RDW = cos(refs_features[0], RDW_features[0]).item()
    similarity_RDW = (similarity_RDW + 1) / 2
    
    image_similarity_CWE.append(similarity_CWE)
    image_similarity_RDW.append(similarity_RDW)
    
    distance_image_CWE = [1 - F for F in image_similarity_CWE]
    distance_image_RDW = [1 - F for F in image_similarity_RDW]

# Save results to files
with open('Results/image_similarity_CWE.npy', 'wb') as f:
    np.save(f, image_similarity_CWE)
with open('Results/image_similarity_RDW.npy', 'wb') as f:
    np.save(f, image_similarity_RDW)
with open('Results/image_distance_CWE.npy', 'wb') as f:
    np.save(f, distance_image_CWE)
with open('Results/image_distance_RDW.npy', 'wb') as f:
    np.save(f, distance_image_RDW)

with open('Results/image_distance_CWE.npy', 'rb') as f1, open('Results/image_distance_RDW.npy', 'rb') as f2:
    a = np.load(f1)
    b = np.load(f2)
print(f"\nCWE: {a}, \nRDW: {b}")

def plot(data, title, y_range, save):
    plt.bar(range(1, 11), data)
    plt.title(title)
    plt.xlabel("Images")
    plt.ylabel("Distance")
    plt.ylim(0, y_range)
    plt.savefig(f"{save}.png", format='png')
    plt.close()
    
# Visualization
plot(distance_F1_CWE, "Differences between two ContextualWordEmbsAug prompts", 0.14, "Results/distant_CWE_prompts")
plot(distance_F1_RDW, "Differences between two RandomWordAug prompts", 0.12, "Results/distant_RDW_prompt")
plot(distance_image_CWE, "Differences between two ContextualWordEmbsAug images", 0.12, "Results/distant_CWE_images")
plot(distance_image_RDW, "Differences between two RandomWordAug images", 0.22, "Results/distant_RDW_images")