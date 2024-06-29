from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def main():
    model_path = "01-ai/Yi-1.5-6B-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    sample_df = get_dataset()
    sample_df['yi-1.5'] = ""

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto',
        max_length=4000,
    ).eval()

    for index, row in tqdm(sample_df.iterrows()):
        input_text = f'Get the most important concepts from this data. Data: {row.text}'
        messages = [
            {"role": "user", "content": input_text}
        ]

        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Model response: "Hello! How can I assist you today?"
        sample_df.at[index, 'yi-1.5'] = response
    sample_df.to_csv('wikihow_with_yi_1_5.csv', index=False)

def get_dataset():
    return pd.read_csv('../wikihowAll_sampled.csv')

if __name__ == "__main__":
    main()
