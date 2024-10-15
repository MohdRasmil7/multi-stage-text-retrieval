from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os

def download_dataset(dataset_name, output_dir):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, output_dir)
    return data_path

def load_dataset(data_path):
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

if __name__ == "__main__":
    dataset_name = "fiqa"  # Or natural_questions, hotpotqa
    output_dir = os.path.join(os.getcwd(), "datasets")
    
    data_path = download_dataset(dataset_name, output_dir)
    corpus, queries, qrels = load_dataset(data_path)
    
    print(f"Corpus loaded: {len(corpus)} documents")
    print(f"Queries loaded: {len(queries)} queries")
