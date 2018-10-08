import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import *


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        """Attend all encoder inputs conditioned on the previous hidden state of the decoder.

        After creating variables to store the attention energies, calculate their
        values for each encoder output and return the normalized values.

        Args:
            hidden: decoder hidden output used for condition
            encoder_outputs: list of encoder outputs

        Returns:
             Normalized (0..1) energy values, re-sized to 1 x 1 x seq_len
        """

        seq_len = len(encoder_outputs)
        energies = Variable(torch.zeros(seq_len)).cuda()
        for i in range(seq_len):
            energies[i] = self._score(hidden, encoder_outputs[i])
        return F.softmax(energies).unsqueeze(0).unsqueeze(0)

    def _score(self, hidden, encoder_output):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = hidden.dot(energy)
        elif self.method == 'concat':
            energy = self.attention(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
        return energy


class AttentionDecoderRNN(nn.Module):
    """Recurrent neural network that makes use of gated recurrent units to translate encoded inputs using attention."""

    def __init__(self, attention_model, hidden_size, output_size, n_layers=1, dropout_p=.1):
        super(AttentionDecoderRNN, self).__init__()
        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attention_model is not None:
            self.attention = Attention(attention_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        """Run forward propagation one step at a time.

        Get the embedding of the current input word (last output word) [s = 1 x batch_size x seq_len]
        then combine them with the previous context. Use this as input and run through the RNN. Next,
        calculate the attention from the current RNN state and all encoder outputs. The final output
        is the next word prediction using the RNN hidden state and context vector.

        Args:
            word_input: torch Variable representing the word input constituent
            last_context: torch Variable representing the previous context
            last_hidden: torch Variable representing the previous hidden state output
            encoder_outputs: torch Variable containing the encoder output values

        Return:
            output: torch Variable representing the predicted word constituent
            context: torch Variable representing the context value
            hidden: torch Variable representing the hidden state of the RNN
            attention_weights: torch Variable retrieved from the attention model
        """

        # Run through RNN
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention
        attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Predict output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, context, hidden, attention_weights


if __name__ == '__main__':
    pass
