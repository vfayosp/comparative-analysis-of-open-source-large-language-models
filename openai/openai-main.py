import pandas as pd
import numpy as np
import time
import os
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

def main():
    client = OpenAI(
                api_key = str(os.environ['OPENAI-KEY'])
            )
    sample_df = get_dataset()
    sample_df['openai'] = ""

    for index, row in tqdm(sample_df.iterrows()):
        input_text = f'Provide the shortest answer. Data: "{row.context}". Question: "{row.question}"'
        messages = [
            {"role": "user", "content": input_text}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Model response: "Hello! How can I assist you today?"
        sample_df.at[index, 'openai'] = response.choices[0].message.content
        sample_df.at[index, 'answers'] = row.answers['text'][0]
    sample_df.to_csv('wikipedia_with_openai_shortest_answer.csv', index=False)

def get_dataset():
    dataset = load_dataset("lhoestq/squad",)
    pd_dataset = dataset['train'].data.to_pandas()

    np.random.seed(44)
    return pd_dataset.sample(n=1000).reset_index(drop=True)

if __name__ == "__main__":
    main()
