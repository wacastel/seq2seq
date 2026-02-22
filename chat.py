import os
import glob
import torch
from datasets import load_dataset
from vocab import prepare_data, normalize_string, tensor_from_sentence, SOS_token, EOS_token
from model import EncoderRNN, DecoderRNN

def evaluate(encoder, decoder, vocab, sentence, max_length=20):
    # Turn off gradients since we are not training
    with torch.no_grad():
        # 1. Process the user's input
        input_normalized = normalize_string(sentence)
        
        # Add a batch dimension of 1
        input_tensor = tensor_from_sentence(vocab, input_normalized).unsqueeze(0) 
        
        # 2. Pass the input through the Encoder
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        # 3. Prepare the Decoder's initial inputs
        decoder_input = torch.tensor([[SOS_token]])  # Start token
        decoder_hidden = encoder_hidden  # Pass the context from the Encoder
        
        decoded_words = []
        
        # 4. Generate the response word by word
        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # Squeeze out the sequence length dimension just like in training
            decoder_output = decoder_output.squeeze(1)
            
            # Get the index of the highest probability word
            topv, topi = decoder_output.topk(1)
            predicted_id = topi.item()
            
            # Stop if the model predicts the End Of Sentence token
            if predicted_id == EOS_token:
                break
            else:
                # Look up the word string from the integer ID
                decoded_words.append(vocab.index2word[predicted_id])
                
            # The predicted word becomes the input for the next time step!
            decoder_input = topi.detach().view(1, 1)
            
        return " ".join(decoded_words)

def get_latest_epoch():
    """Finds the highest epoch number from the saved checkpoint files."""
    encoder_files = glob.glob("encoder_epoch_*.pth")
    if not encoder_files:
        return None
    
    epochs = []
    for f in encoder_files:
        try:
            # Extract the integer from the filename (e.g., "encoder_epoch_10.pth" -> 10)
            epoch = int(f.split('_')[-1].split('.')[0])
            epochs.append(epoch)
        except ValueError:
            continue
            
    return max(epochs) if epochs else None

def chat_with_bot():
    latest_epoch = get_latest_epoch()
    
    if latest_epoch is None:
        print("No checkpoint files found! Please wait for train.py to finish an epoch.")
        return

    encoder_path = f"encoder_epoch_{latest_epoch}.pth"
    decoder_path = f"decoder_epoch_{latest_epoch}.pth"

    print(f"Found checkpoints from epoch {latest_epoch}!")
    print("Loading vocabulary...")
    # Rebuild the vocabulary so we have our integer-to-word mappings
    dataset = load_dataset("AlekseyKorshuk/persona-chat")
    vocab, _ = prepare_data(dataset['train'])
    
    hidden_size = 256
    print(f"Loading saved models ({encoder_path} and {decoder_path})...")
    encoder = EncoderRNN(vocab.num_words, hidden_size)
    decoder = DecoderRNN(hidden_size, vocab.num_words)
    
    # Load the saved weights
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    
    # Set models to evaluation mode (disables dropout layers for consistent predictions)
    encoder.eval()
    decoder.eval()
    
    print("\n" + "="*40)
    print("Chatbot is ready! Type 'quit' to exit.")
    print("="*40)
    
    # The Chat Loop!
    while True:
        user_input = input("> You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        response = evaluate(encoder, decoder, vocab, user_input)
        print(f"> Bot: {response}")

if __name__ == "__main__":
    chat_with_bot()
