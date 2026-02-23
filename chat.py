import os
import glob
import torch
import torch.nn.functional as F
from datasets import load_dataset
from vocab import prepare_data, normalize_string, tensor_from_sentence, SOS_token, EOS_token
from model import EncoderRNN, AttnDecoderRNN

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

# 1. Added a 'temperature' parameter (defaulting to 0.7 is a standard practice)
def evaluate(encoder, decoder, vocab, persona, sentence, max_length=20, temperature=0.7):
    with torch.no_grad():
        input_text = f"{persona} {sentence}"
        input_normalized = normalize_string(input_text)
        
        input_tensor = tensor_from_sentence(vocab, input_normalized).unsqueeze(0) 
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        decoder_input = torch.tensor([[SOS_token]]) 
        decoder_hidden = encoder_hidden 
        
        decoded_words = []
        
        for _ in range(max_length):
            # Your decoder outputs raw "logits" (un-normalized scores)
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_output = decoder_output.squeeze(1)
            
            # --- NEW: Temperature Scaling & Sampling ---
            # 1. Scale the logits by the temperature
            scaled_logits = decoder_output / temperature
            
            # 2. Convert the scaled logits into a probability distribution (values between 0 and 1)
            probs = torch.softmax(scaled_logits, dim=1)
            
            # 3. Roll the dice! Sample 1 word randomly based on its calculated probability
            predicted_id = torch.multinomial(probs, 1).item()
            # -------------------------------------------
            
            if predicted_id == EOS_token:
                break
            else:
                decoded_words.append(vocab.index2word[predicted_id])
                
            # Convert the chosen integer ID back into a tensor for the next time step
            decoder_input = torch.tensor([[predicted_id]])
            
        return " ".join(decoded_words)

def chat_with_bot():
    latest_epoch = get_latest_epoch()
    if latest_epoch is None:
        print("No checkpoints found!")
        return

    encoder_path = f"encoder_epoch_{latest_epoch}.pth"
    decoder_path = f"decoder_epoch_{latest_epoch}.pth"

    print("Loading vocabulary...")
    dataset = load_dataset("AlekseyKorshuk/persona-chat")
    vocab, _ = prepare_data(dataset['train'])
    
    hidden_size = 256
    encoder = EncoderRNN(vocab.num_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, vocab.num_words)
    
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
    
    encoder.eval()
    decoder.eval()
    
    # --- Define your bot's personality here! ---
    bot_persona = (
        "i am a software engineer . "
        "i collect vintage apple computers . "
        "i love playing retro video games . "
        "i am a fan of nine inch nails ."
    )
    
    print("\n" + "="*50)
    print(f"Bot Persona: {bot_persona}")
    print("Chatbot is ready! Type 'quit' to exit.")
    print("="*50)
    
    while True:
        user_input = input("> You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        # 3. Pass the persona into the evaluate function
        response = evaluate(encoder, decoder, vocab, bot_persona, user_input)
        print(f"> Bot: {response}")

if __name__ == "__main__":
    chat_with_bot()
