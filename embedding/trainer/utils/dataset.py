import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, data, max_seq_length=50):
        self.data = data
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]  # Assuming 'data' is a list of strings
        return text

# Custom collate function to handle variable-length sequences
def collate_fn(batch):
    # Tokenize or preprocess your text data as needed
    # Here, we simply split the text into words
    tokenized_batch = [text.split() for text in batch]

    # Convert words to indices
    indexed_batch = [torch.tensor([word_to_index[word] for word in tokens]) for tokens in tokenized_batch]

    # Pad sequences to the maximum length in the batch
    padded_batch = pad_sequence(indexed_batch, batch_first=True, padding_value=0)

    return padded_batch

# Example usage
data = ["This is an example.", "Another sentence for demonstration.", "Short text."]
dataset = TextDataset(data)

# Build vocabulary (word_to_index dictionary)
word_to_index = {}
for idx, word in enumerate(set(word for text in data for word in text.split())):
    word_to_index[word] = idx + 1  # Start index from 1, leaving 0 for padding

# Create an embedding layer
embedding_dim = 100
embedding = Embedding(len(word_to_index) + 1, embedding_dim)

# Create DataLoader
batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# Example usage in a training loop
for batch in dataloader:
    embedded_batch = embedding(batch)
    print(embedded_batch)