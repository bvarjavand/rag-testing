import pandas as pd
import datasets

class DataLoader:
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split
    
    def load_prompts(self, limit=None):
        dataset = datasets.load_dataset(self.dataset_name)[self.split]
        prompt_df = dataset.to_pandas()
        prompt_df['text'] = prompt_df['prompt'].apply(lambda x: x['text'] + x['continuation']['text'])
        return prompt_df.head(limit)
    
    def load_documents(self):
        # Load documents to be stored in Weaviate
        pass  # Add dataset loading and processing implementation