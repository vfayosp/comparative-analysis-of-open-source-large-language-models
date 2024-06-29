from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def main():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
    
    sample_df = get_dataset()
    sample_df['flan-t5'] = ""

    for index, row in tqdm(sample_df.iterrows()):
        input_text = f'Get the most important concepts from this data. Data: {row.text}'
        input_ids = tokenizer(input_text,max_length=4096, return_tensors="pt").input_ids

        outputs = model.generate(input_ids)
        sample_df.at[index, 'flan-t5'] = tokenizer.decode(outputs[0])
    sample_df.to_csv('wikihow_with_flan_t5.csv', index=False)

def get_dataset():
    return pd.read_csv('../wikihowAll_sampled.csv')

if __name__ == "__main__":
    main()
