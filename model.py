import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Converts word indices into dense vectors
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        
        # The GRU layer processes the sequence
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_seq):
        # input_seq shape: (batch_size, sequence_length)
        embedded = self.dropout(self.embedding(input_seq))
        
        # output contains the hidden states for each time step
        # hidden contains the final state for the entire sequence
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Converts predicted word indices back into dense vectors for the next step
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        
        # The GRU layer generates the next hidden state
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Linear layer maps the hidden state to a probability distribution over the vocabulary
        self.out = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, input_token, hidden):
        # input_token shape: (batch_size, 1) - processing one word at a time
        embedded = self.embedding(input_token)
        
        # Pass the current token and the previous hidden state to the GRU
        output, hidden = self.gru(embedded, hidden)
        
        # Predict the next word in the sequence
        prediction = self.out(output)
        
        return prediction, hidden

class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongAttention, self).__init__()
        # A linear layer to transform the encoder outputs before the dot product
        self.Wa = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden shape: (batch_size, 1, hidden_size)
        # encoder_outputs shape: (batch_size, sequence_length, hidden_size)
        
        # 1. Transform the encoder outputs
        transformed_encoder_outputs = self.Wa(encoder_outputs)
        
        # 2. Calculate alignment scores using a dot product (batch matrix multiplication)
        # We transpose the decoder_hidden to align the dimensions for bmm
        scores = torch.bmm(transformed_encoder_outputs, decoder_hidden.transpose(1, 2))
        # scores shape: (batch_size, sequence_length, 1)

        # 3. Apply Softmax to get the attention weights
        attn_weights = F.softmax(scores, dim=1)
        
        # 4. Multiply weights by encoder outputs to get the weighted context vector
        context_vector = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        # context_vector shape: (batch_size, 1, hidden_size)
        
        return context_vector, attn_weights

# Assuming LuongAttention is defined above this in the same file

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_vocab_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_vocab_size = output_vocab_size
        
        # 1. Embedding and Dropout
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
        # 2. The GRU Layer
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # 3. The Attention Module
        self.attention = LuongAttention(hidden_size)
        
        # 4. Linear layers to combine the context vector and GRU output
        # We multiply by 2 because we are concatenating the GRU hidden state and the context vector
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token shape: (batch_size, 1)
        # encoder_outputs shape: (batch_size, sequence_length, hidden_size)
        
        # Get the embedding of the current input word
        embedded = self.dropout(self.embedding(input_token))
        
        # Pass the embedding and previous hidden state through the GRU
        rnn_output, hidden = self.gru(embedded, hidden)
        # rnn_output shape: (batch_size, 1, hidden_size)
        
        # Calculate the attention weights and the resulting context vector
        context_vector, attn_weights = self.attention(rnn_output, encoder_outputs)
        # context_vector shape: (batch_size, 1, hidden_size)
        
        # Concatenate the GRU output and the attention context vector along the feature dimension
        concat_input = torch.cat((rnn_output, context_vector), dim=2)
        # concat_input shape: (batch_size, 1, hidden_size * 2)
        
        # Pass the concatenated vector through a linear layer and a tanh activation
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Predict the next word probability distribution
        prediction = self.out(concat_output)
        # prediction shape: (batch_size, 1, output_vocab_size)
        
        return prediction, hidden, attn_weights
