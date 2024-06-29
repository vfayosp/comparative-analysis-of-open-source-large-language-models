from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import pandas as pd
import numpy as np
import time

def main():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
    
    sample_df = get_dataset()
    sample_df['flan-t5'] = ""

    for index, row in sample_df.iterrows():
        if index%100==0: print(f'Going through iteration {index}')

        input_text = f'Provide the shortest answer. Data: "{row.context}". Question: "{row.question}"'
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        outputs = model.generate(input_ids)
        sample_df.at[index, 'flan-t5'] = tokenizer.decode(outputs[0])
        sample_df.at[index, 'answers'] = row.answers['text'][0]
    sample_df.to_csv('wikipedia_with_flan_t5_shortest_answer.csv', index=False)

def get_dataset():
    dataset = load_dataset("lhoestq/squad",)
    pd_dataset = dataset['train'].data.to_pandas()

    np.random.seed(44)
    return pd_dataset.sample(n=1000).reset_index(drop=True)

if __name__ == "__main__":
    main()
