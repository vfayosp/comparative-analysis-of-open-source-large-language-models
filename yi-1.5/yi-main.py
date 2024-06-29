from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import numpy as np
import time

def main():
    model_path = "01-ai/Yi-1.5-6B-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    sample_df = get_dataset()
    sample_df['yi-1.5'] = ""

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto',
        max_length=2000,
    ).eval()

    for index, row in sample_df.iterrows():
        if index%100==0: print(f'Going through iteration {index}')
        input_text = f'Provide the shortest answer. Data: "{row.context}". Question: "{row.question}"'
        messages = [
            {"role": "user", "content": input_text}
        ]

        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'), eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Model response: "Hello! How can I assist you today?"
        sample_df.at[index, 'yi-1.5'] = response
        sample_df.at[index, 'answers'] = row.answers['text'][0]
    sample_df.to_csv('wikipedia_with_yi_1_5_shortest_answer.csv', index=False)

def get_dataset():
    dataset = load_dataset("lhoestq/squad",)
    pd_dataset = dataset['train'].data.to_pandas()

    np.random.seed(44)
    return pd_dataset.sample(n=1000).reset_index(drop=True)

if __name__ == "__main__":
    main()
