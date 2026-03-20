from datasets import Dataset

def build_dataset():
    texts = [
        "Physics explains natural phenomena using mathematical models.",
        "Machine learning models learn patterns from data.",
        "Transformers process sequences using self-attention.",
    ]
    return Dataset.from_dict({"text": texts})