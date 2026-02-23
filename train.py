import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datasets import load_dataset
from tqdm import tqdm

from vocab import prepare_data, prepare_batch, SOS_token, PAD_token
from model import EncoderRNN, AttnDecoderRNN

# Import the evaluate function you already wrote in your chat script!
from chat import evaluate  

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor)

    batch_size = input_tensor.size(0)
    decoder_input = torch.tensor([[SOS_token]] * batch_size)
    decoder_hidden = encoder_hidden

    loss = 0
    target_length = target_tensor.size(1)

    for t in range(target_length):
        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
        
        decoder_output = decoder_output.squeeze(1)
        loss += criterion(decoder_output, target_tensor[:, t])
        decoder_input = target_tensor[:, t].unsqueeze(1)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def get_latest_epoch():
    """Finds the highest epoch number from the saved checkpoint files."""
    encoder_files = glob.glob("encoder_epoch_*.pth")
    if not encoder_files:
        return None
    
    epochs = []
    for f in encoder_files:
        try:
            epoch = int(f.split('_')[-1].split('.')[0])
            epochs.append(epoch)
        except ValueError:
            continue
            
    return max(epochs) if epochs else None

def evaluate_randomly(encoder, decoder, vocab, pairs, n=2):
    """Picks random training pairs and prints the bot's prediction vs the actual target."""
    # Temporarily set models to evaluation mode to disable dropout
    encoder.eval()
    decoder.eval()
    
    for i in range(n):
        pair = random.choice(pairs)
        print(f"> Input:  {pair[0]}")
        print(f"= Target: {pair[1]}")
        
        # Use the evaluate function to generate a response
        output_words = evaluate(encoder, decoder, vocab, pair[0])
        print(f"< Bot:    {output_words}\n")
        
    # Set models back to training mode
    encoder.train()
    decoder.train()

def train_epochs(epochs, batch_size=32, hidden_size=256, learning_rate=0.001):
    print("Loading data...")
    dataset = load_dataset("AlekseyKorshuk/persona-chat")
    vocab, training_pairs = prepare_data(dataset['train'])
    
    print("Initializing models...")
    encoder = EncoderRNN(input_vocab_size=vocab.num_words, hidden_size=hidden_size)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, output_vocab_size=vocab.num_words)
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    # --- RESUME TRAINING LOGIC ---
    start_epoch = 1
    latest_epoch = get_latest_epoch()
    
    if latest_epoch is not None:
        print(f"Found checkpoints from epoch {latest_epoch}. Resuming training!")
        encoder.load_state_dict(torch.load(f"encoder_epoch_{latest_epoch}.pth", weights_only=True))
        decoder.load_state_dict(torch.load(f"decoder_epoch_{latest_epoch}.pth", weights_only=True))
        
        # Start at the next subsequent epoch
        start_epoch = latest_epoch + 1

    # Check if we have already reached the target number of epochs
    if start_epoch > epochs:
        print(f"Model already trained to {latest_epoch} epochs. Increase the 'epochs' argument to train further.")
        return

    print(f"Starting training from epoch {start_epoch} to {epochs}...")
    
    for epoch in range(start_epoch, epochs + 1):
        random.shuffle(training_pairs)
        epoch_loss = 0
        
        batch_iterator = tqdm(range(0, len(training_pairs), batch_size), 
                              desc=f"Epoch {epoch}/{epochs}", 
                              unit="batch")
        
        for i in batch_iterator:
            batch_pairs = training_pairs[i:i+batch_size]
            input_tensors, target_tensors = prepare_batch(vocab, batch_pairs)
            
            loss = train_step(input_tensors, target_tensors, encoder, decoder, 
                              encoder_optimizer, decoder_optimizer, criterion)
            
            epoch_loss += loss
            batch_iterator.set_postfix(loss=f"{loss:.4f}")
                
        print(f"\n--- Epoch {epoch} Complete | Average Loss: {epoch_loss / (len(training_pairs) / batch_size):.4f} ---")
        
        # --- RANDOM EVALUATION ---
        print("\n--- Random Evaluation ---")
        evaluate_randomly(encoder, decoder, vocab, training_pairs, n=2)
        
        checkpoint_name_enc = f"encoder_epoch_{epoch}.pth"
        checkpoint_name_dec = f"decoder_epoch_{epoch}.pth"
        print(f"Saving checkpoint to {checkpoint_name_enc} and {checkpoint_name_dec}...")
        torch.save(encoder.state_dict(), checkpoint_name_enc)
        torch.save(decoder.state_dict(), checkpoint_name_dec)
        
    print("\nTraining completely finished!")

if __name__ == "__main__":
    train_epochs(epochs=10, batch_size=64)
