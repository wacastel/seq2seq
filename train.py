import torch
import torch.nn as nn
import torch.optim as optim
import random
from datasets import load_dataset
from tqdm import tqdm  # Add this import

# Import your custom modules
from vocab import prepare_data, prepare_batch, SOS_token, PAD_token
from model import EncoderRNN, DecoderRNN

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # 1. Clear previous gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 2. Forward pass through the Encoder
    # input_tensor shape: (batch_size, sequence_length)
    encoder_outputs, encoder_hidden = encoder(input_tensor)

    # 3. Prepare initial Decoder inputs
    batch_size = input_tensor.size(0)
    
    # The first token for every sequence in the batch is the <SOS> token
    decoder_input = torch.tensor([[SOS_token]] * batch_size)
    
    # The initial hidden state of the Decoder is the final hidden state of the Encoder
    decoder_hidden = encoder_hidden

    loss = 0
    target_length = target_tensor.size(1)

    # 4. Teacher Forcing Loop: Pass the actual target words as the next input
    for t in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        
        # FIX: Remove the sequence length dimension (dim 1) 
        # Changes shape from (batch_size, 1, vocab_size) to (batch_size, vocab_size)
        decoder_output = decoder_output.squeeze(1)
        
        # Calculate loss for this specific time step
        loss += criterion(decoder_output, target_tensor[:, t])
        
        # Teacher forcing: the next input is the current ground-truth target word
        decoder_input = target_tensor[:, t].unsqueeze(1)

    # 5. Backpropagation and Optimization
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return the average loss per token for this batch
    return loss.item() / target_length

def train_epochs(epochs, batch_size=32, hidden_size=256, learning_rate=0.001):
    print("Loading data...")
    dataset = load_dataset("AlekseyKorshuk/persona-chat")
    vocab, training_pairs = prepare_data(dataset['train'])
    
    print("Initializing models...")
    # Instantiate the Encoder and Decoder
    encoder = EncoderRNN(input_vocab_size=vocab.num_words, hidden_size=hidden_size)
    decoder = DecoderRNN(hidden_size=hidden_size, output_vocab_size=vocab.num_words)
    
    # Set up Optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    # Set up the Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)

    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Shuffle the data at the start of each epoch
        random.shuffle(training_pairs)
        epoch_loss = 0
        
        # Wrap our range() in tqdm to create the progress bar
        batch_iterator = tqdm(range(0, len(training_pairs), batch_size), 
                              desc=f"Epoch {epoch}/{epochs}", 
                              unit="batch")
        
        # Iterate through the data using the progress bar
        for i in batch_iterator:
            batch_pairs = training_pairs[i:i+batch_size]
            
            # Convert text to padded tensors
            input_tensors, target_tensors = prepare_batch(vocab, batch_pairs)
            
            # Run the training step
            loss = train_step(input_tensors, target_tensors, encoder, decoder, 
                              encoder_optimizer, decoder_optimizer, criterion)
            
            epoch_loss += loss
            
            # Update the progress bar to show the current loss in real-time
            batch_iterator.set_postfix(loss=f"{loss:.4f}")
                
        print(f"\n--- Epoch {epoch} Complete | Average Loss: {epoch_loss / (len(training_pairs) / batch_size):.4f} ---")
        
        # --- NEW CHECKPOINTING LOGIC ---
        # Save a unique file for each epoch so you never lose progress
        checkpoint_name_enc = f"encoder_epoch_{epoch}.pth"
        checkpoint_name_dec = f"decoder_epoch_{epoch}.pth"
        
        print(f"Saving checkpoint to {checkpoint_name_enc} and {checkpoint_name_dec}...")
        torch.save(encoder.state_dict(), checkpoint_name_enc)
        torch.save(decoder.state_dict(), checkpoint_name_dec)
        
    print("\nTraining completely finished!")

if __name__ == "__main__":
    # Bumped the epochs up to 10 for a proper training run!
    train_epochs(epochs=10, batch_size=64)

