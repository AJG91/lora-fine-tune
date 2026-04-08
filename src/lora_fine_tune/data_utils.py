from datasets import Dataset

def build_dataset():
    texts = [
        "Q: How do transformers work?\nA: Transformers process sequences using self-attention.",
        "Q: What does physics explain?\nA: Physics explains natural phenomena using mathematical models.",
        "Q: What do machine learning models do?\nA: Machine learning models learn patterns from data.",
        "Q: How do transformers work?\nA: Transformers process sequences using self-attention mechanisms.",
        "Q: What is self-attention?\nA: Self-attention allows tokens in a sequence to attend to each other.",
        "Q: What is a neural network?\nA: A neural network is a model composed of layers of interconnected nodes.",
    ]
    return Dataset.from_dict({"text": texts})