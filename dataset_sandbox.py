from datasets import load_dataset
from vocab import prepare_data, prepare_batch

def main():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("AlekseyKorshuk/persona-chat")
    
    # 1. Build the vocabulary and extract the text pairs
    vocab, training_pairs = prepare_data(dataset['train'])
    
    # 2. Grab a small "batch" of 4 examples to test our tensor conversion
    sample_batch = training_pairs[:4]
    
    # 3. Convert the text batch into padded tensors
    padded_inputs, padded_targets = prepare_batch(vocab, sample_batch)
    
    print("\n--- Padded Input Tensor Shape ---")
    print(padded_inputs.shape) # Expected: (4, longest_input_length)
    
    print("\n--- First Input Tensor in Batch ---")
    print(padded_inputs[0]) # You will see the integer IDs ending with 2 (EOS) and 0s (PAD)

if __name__ == "__main__":
    main()
