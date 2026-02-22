import re
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
# Make sure to import your normalize_string and Vocabulary from the previous step

# Standard special tokens for Seq2Seq models
PAD_token = 0  # Used to pad sequences to the same length in a batch
SOS_token = 1  # Start Of Sequence: tells the decoder to start generating
EOS_token = 2  # End Of Sequence: tells the decoder to stop generating
UNK_token = 3  # Unknown word: used if we encounter a word not in our vocabulary

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        # Pre-populate with our special tokens
        self.index2word = {
            PAD_token: "<PAD>", 
            SOS_token: "<SOS>", 
            EOS_token: "<EOS>", 
            UNK_token: "<UNK>"
        }
        self.num_words = 4  # Count starts at 4 because of the special tokens

    def add_sentence(self, sentence):
        """Splits a normalized sentence and adds each word to the vocab."""
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """Adds a single word to the vocabulary dictionaries."""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

def normalize_string(s):
    """
    Lowercases the string, isolates punctuation, and removes non-alphabetic characters.
    """
    s = s.lower().strip()
    # Put a space before punctuation so it's treated as a separate token
    s = re.sub(r"([.!?])", r" \1", s)
    # Replace anything that isn't a letter or basic punctuation with a space
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def prepare_data(dataset_split, vocab_name="persona_vocab"):
    print(f"Reading and processing {len(dataset_split)} examples...")
    
    vocab = Vocabulary(vocab_name)
    pairs = []
    
    for example in dataset_split:
        # 1. Combine the personality traits into a single background context string
        personality_context = " ".join(example['personality'])
        
        for utterance in example['utterances']:
            # 2. Combine the conversation history
            history_context = " ".join(utterance['history'])
            
            # 3. Create the full input text for the Encoder
            input_text = f"{personality_context} {history_context}"
            
            # 4. The true response is the last item in the candidates list
            target_text = utterance['candidates'][-1]
            
            # 5. Normalize the strings to remove weird characters and fix punctuation
            input_normalized = normalize_string(input_text)
            target_normalized = normalize_string(target_text)
            
            # 6. Add the words to our Vocabulary tracking
            vocab.add_sentence(input_normalized)
            vocab.add_sentence(target_normalized)
            
            # 7. Save the pair for training
            pairs.append((input_normalized, target_normalized))
            
    print(f"Counted {vocab.num_words} unique words in the vocabulary.")
    print(f"Created {len(pairs)} input-target training pairs.")
    
    return vocab, pairs

# Assuming PAD_token = 0 and EOS_token = 2 were defined at the top of vocab.py

def indexes_from_sentence(vocab, sentence):
    """Converts a normalized string into a list of integer word IDs."""
    # We add a quick check to ensure we only grab words that exist in the vocab
    return [vocab.word2index[word] for word in sentence.split(' ') if word in vocab.word2index]

def tensor_from_sentence(vocab, sentence):
    """Converts a list of integer IDs into a PyTorch tensor and appends the EOS token."""
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(EOS_token) # Tell the network the sentence is over
    # dtype=torch.long is required for PyTorch embedding layers
    return torch.tensor(indexes, dtype=torch.long)

def prepare_batch(vocab, batch_pairs):
    """
    Takes a list of (input, target) text pairs, converts them to tensors, 
    and pads them so they all have the same length in the batch.
    """
    input_tensors = []
    target_tensors = []
    
    for input_text, target_text in batch_pairs:
        input_tensors.append(tensor_from_sentence(vocab, input_text))
        target_tensors.append(tensor_from_sentence(vocab, target_text))
        
    # pad_sequence automatically finds the longest tensor and pads the rest to match.
    # batch_first=True makes the output tensor shape (batch_size, max_sequence_length)
    padded_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=PAD_token)
    padded_targets = pad_sequence(target_tensors, batch_first=True, padding_value=PAD_token)
    
    return padded_inputs, padded_targets

# --- How to use it together ---
if __name__ == "__main__":
    print("Loading dataset from Hugging Face...")
    raw_dataset = load_dataset("AlekseyKorshuk/persona-chat")
    
    # Process the training split
    vocab, training_pairs = prepare_data(raw_dataset['train'])
    
    # Let's peek at one of the processed pairs!
    print("\n--- Example Processed Pair ---")
    print("INPUT:", training_pairs[0][0])
    print("TARGET:", training_pairs[0][1])
