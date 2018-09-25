import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim

from config import *
from data_generator import TranslationDataset
from models import EncoderRNN, AttnDecoderRNN

plt.switch_backend('agg')
import matplotlib.ticker as ticker


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()
    print_loss_total = 0  # Reset every print_every

    # Batches
    for i, (input_tensor, target_tensor) in enumerate(train_loader):
        start = time.time()

        # Move to GPU, if available
        input_tensor = input_tensor.view(-1, 1).to(device)
        target_tensor = target_tensor.view(-1, 1).to(device)
        # print('input_tensor: ' + str(input_tensor))
        print('target_tensor: ' + str(target_tensor))
        # print('input_tensor.size(): ' + str(input_tensor.size()))
        print('target_tensor.size(): ' + str(target_tensor.size()))

        input_length = np.where(input_tensor.cpu().numpy() == EOS_token)[0][0]
        target_length = np.where(target_tensor.cpu().numpy() == EOS_token)[0][0]
        print('input_length: ' + str(input_length))
        print('target_length: ' + str(target_length))

        encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        # Back prop.
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()
        print_loss_total += loss.item()

        encoder_optimizer.step()
        decoder_optimizer.step()

        if (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, (i + 1) / num_train_samples),
                                         num_train_samples, (i + 1) / num_train_samples * 100, print_loss_avg))

    return loss.item() / target_length


def trainIters(train_loader, encoder, decoder, learning_rate=0.01):
    plot_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Epochs
    for epoch in range(start_epoch, epochs):
        loss = train(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        plot_losses.append(loss)

    showPlot(plot_losses)


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset('train'), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        TranslationDataset('valid'), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    encoder = EncoderRNN(input_lang_n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang_n_words, dropout_p=0.1).to(device)

    trainIters(train_loader, encoder, attn_decoder)
