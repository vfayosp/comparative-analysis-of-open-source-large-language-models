import pandas as pd
import numpy as np
import time
import os
from openai import OpenAI
from tqdm import tqdm

def main():
    client = OpenAI(
                api_key = str(os.environ['OPENAI-KEY'])
            )
    sample_df = get_dataset()
    sample_df['openai'] = ""

    for index, row in tqdm(sample_df.iterrows()):
        input_text = f'Get the most important concepts from this data. Data: {row.text}'
        messages = [
            {"role": "user", "content": input_text}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Model response: "Hello! How can I assist you today?"
        sample_df.at[index, 'openai'] = response.choices[0].message.content
    sample_df.to_csv('wikihow_with_openai.csv', index=False)

def get_dataset():
    return pd.read_csv('../wikihowAll_sampled.csv')

if __name__ == "__main__":
    main()
