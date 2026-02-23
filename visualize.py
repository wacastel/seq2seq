import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from datasets import load_dataset

from vocab import prepare_data, normalize_string, tensor_from_sentence, SOS_token, EOS_token
from model import EncoderRNN, AttnDecoderRNN
from chat import get_latest_epoch

def evaluate_and_get_attention(encoder, decoder, vocab, persona, sentence, max_length=20):
    """
    Runs inference but explicitly captures and returns the Attention weights 
    generated at every single time step.
    """
    with torch.no_grad():
        input_text = f"{persona} {sentence}"
        input_normalized = normalize_string(input_text)
        
        input_tensor = tensor_from_sentence(vocab, input_normalized).unsqueeze(0) 
        
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        decoder_input = torch.tensor([[SOS_token]]) 
        decoder_hidden = encoder_hidden 
        
        decoded_words = []
        decoder_attentions = [] # Array to store the attention weights
        
        for _ in range(max_length):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Save the attention weights for this specific step!
            # Shape goes from (1, 1, seq_len) -> (seq_len,)
            decoder_attentions.append(attn_weights.squeeze().cpu().numpy())
            
            decoder_output = decoder_output.squeeze(1)
            topv, topi = decoder_output.topk(1)
            predicted_id = topi.item()
            
            if predicted_id == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(vocab.index2word[predicted_id])
                
            decoder_input = topi.detach().view(1, 1)
            
        return decoded_words, decoder_attentions, input_normalized

def show_attention(input_sentence, output_words, attentions):
    """Generates a heatmap of the attention weights using matplotlib."""
    # Set up figure with a colorbar
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Convert list of arrays into a 2D numpy matrix
    attention_matrix = np.array(attentions)
    
    # Render the heatmap using a visually distinct colormap (viridis)
    cax = ax.matshow(attention_matrix, cmap='viridis')
    fig.colorbar(cax)

    # Set up the axes labels
    input_tokens = input_sentence.split(' ') + ['<EOS>']
    
    # Note: We prepend empty strings because Matplotlib's ticker starts indexing at 1 internally
    ax.set_xticklabels([''] + input_tokens, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Save the plot instead of trying to open a window
    output_filename = 'attention_heatmap.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nSuccess! Attention heatmap saved as: {output_filename}")

def main():
    latest_epoch = get_latest_epoch()
    if latest_epoch is None:
        print("No checkpoints found! Please train the model first.")
        return

    print("Loading vocabulary and models...")
    dataset = load_dataset("AlekseyKorshuk/persona-chat")
    vocab, _ = prepare_data(dataset['train'])
    
    hidden_size = 256
    encoder = EncoderRNN(vocab.num_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, vocab.num_words)
    
    encoder.load_state_dict(torch.load(f"encoder_epoch_{latest_epoch}.pth", weights_only=True))
    decoder.load_state_dict(torch.load(f"decoder_epoch_{latest_epoch}.pth", weights_only=True))
    
    encoder.eval()
    decoder.eval()
    
    # The persona and test sentence
    bot_persona = (
        "i am a software engineer . "
        "i collect vintage apple computers . "
        "i love playing retro video games . "
        "i am a fan of nine inch nails ."
    )
    user_input = "What kind of computers do you like to collect?"
    
    print("\n--- Generating Response ---")
    print(f"User: {user_input}")
    
    output_words, attentions, input_normalized = evaluate_and_get_attention(
        encoder, decoder, vocab, bot_persona, user_input
    )
    
    print(f"Bot: {' '.join(output_words)}")
    
    # Generate the plot
    show_attention(input_normalized, output_words, attentions)

if __name__ == "__main__":
    main()
