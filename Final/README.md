## Data Download

You can download the dataset by clicking on the following link:
[Download Link](https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-000001.zip?download=true)

## Dependency

```bash
pip install openai
pip install python-dotenv
pip install numpy requests nlpaug
pip install nltk
pip install torch   
pip install bert-score
pip install git+https://github.com/openai/CLIP.git
```

## Create '.env' File

Create a `.env` file in your project directory and add the following lines. Replace the placeholder values with your actual data:

```plaintext
OPENAI_API_KEY=''
IMAGE_FILE_PATH=''
NUM_PROMPTS=10
```

## How to Run

```bash
cd .\Final\
python generate_image.py
python compute_difference.py
```