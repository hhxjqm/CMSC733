## Data Download

You can download the dataset by clicking on the following link:
[Download Link](https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-000001.zip?download=true)

## Create '.env' File

Create a `.env` file in your project directory and add the following lines. Replace the placeholder values with your actual data:

```plaintext
OPENAI_API_KEY=''
IMAGE_FILE_PATH=''
NUM_PROMPTS=10
```

## How to Run

```bash
python generate_image.py
```