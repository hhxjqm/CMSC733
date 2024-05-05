from openai import OpenAI
import openai
import json
import random
from dotenv import load_dotenv
import os
import shutil
import requests
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import os
import json
load_dotenv()

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def select_random_prompts(data, num_prompts=10):
    if len(data) > num_prompts:
        selected_keys = random.sample(list(data.keys()), num_prompts)
    else:
        selected_keys = list(data.keys())
    selected_prompts = {key: data[key]['p'] for key in selected_keys}
    return selected_prompts

def find_and_copy_file(source_folder, target_folder, filename):
    for root, dirs, files in os.walk(source_folder):
        if filename in files:
            file_path = os.path.join(root, filename)
            target_path = os.path.join(target_folder, filename)
            shutil.copy(file_path, target_path)
            break
    else:
        print("Image not exit")
        
def download_image(image_url, save_path):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, file)
        # print(f"Image saved as: {save_path}")
    else:
        print("Download error:", response.status_code)

def create_directory(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def augmenter_ContextualWordEmbsAug(text):    
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    augmented_text = aug.augment(text)
    return augmented_text

def augmenter_RandomWordAug(text):    
    aug = naw.RandomWordAug()
    augmented_text = aug.augment(text)
    return augmented_text

def get_url(client,prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

def main():
    file_path = 'part-000001.json' 
    source_folder = os.getenv('IMAGE_FILE_PATH')
    num_prompts = int(os.getenv('NUM_PROMPTS'))
    data = load_json_data(file_path)
    random_prompts = select_random_prompts(data, num_prompts)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    dict = {}
    
    i = 11
    for image_name, prompt in random_prompts.items():
        print(f"\n===============================Image{i}============================================\n")
        save_path = f'new_image/image{i}'
        try:
            # Prepare 
            create_directory(save_path)
            find_and_copy_file(source_folder, save_path, image_name)
            
            # Get new prompt
            new_prompt_ContextualWordEmbsAug = augmenter_ContextualWordEmbsAug(prompt)[0]
            new_prompt_RandomWordAug = augmenter_RandomWordAug(prompt)[0]
            
            # Get new image
            image_url_origin = get_url(client, prompt)
            image_url_ContextualWordEmbsAug = get_url(client, new_prompt_ContextualWordEmbsAug)
            image_url_RandomWordAug = get_url(client, new_prompt_RandomWordAug)
            
            # Save new image
            download_image(image_url_origin, f'{save_path}/origin.webp')
            download_image(image_url_ContextualWordEmbsAug, f'{save_path}/ContextualWordEmbsAug.webp')
            download_image(image_url_RandomWordAug, f'{save_path}/RandomWordAug.webp')
            
            # Print prompts
            print(f"Origin:                {prompt} \n")
            print(f"ContextualWordEmbsAug: {new_prompt_ContextualWordEmbsAug} \n")
            print(f"RandomWordAug:         {new_prompt_RandomWordAug}")
            
            # Save
            dict[f"image{i}"] = {
                "p": prompt,
                "CWE_p": new_prompt_ContextualWordEmbsAug,
                "RDW_p": new_prompt_RandomWordAug,
                "path": f"new_image/image{i}",
                "b_img": image_name
            }
        except openai.BadRequestError as e:
            print("Rejected from openai safety system, prompt may contain text that is not allowed")
            i = i + 1
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        i = i + 1
    with open('Results/data.json', 'w') as file:
        json.dump(dict, file, indent=4)
if __name__ == '__main__':
    main()