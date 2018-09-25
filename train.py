import math
import random
import time

import matplotlib.pyplot as plt
from torch import nn
from torch import optim

from config import *
from data_generator import TranslationDataset
from models import EncoderRNN, AttnDecoderRNN
from utils import AverageMeter

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


def calc_loss(input_tensor, input_length, target_tensor, target_length, encoder, decoder, criterion):
    encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)
    encoder_hidden = encoder.initHidden()

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    loss = 0
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

    return loss / target_length


def train(epoch, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    print_loss_total = 0  # Reset every print_every

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)

    # Batches
    for i, (input_array, target_array) in enumerate(train_loader):
        start = time.time()
        # Move to GPU, if available
        input_tensor = torch.tensor(input_array, device=device).view(-1, 1)
        target_tensor = torch.tensor(target_array, device=device).view(-1, 1)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        loss = calc_loss(input_tensor, input_length, target_tensor, target_length, encoder, decoder, criterion)

        # Back prop.
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()
        print_loss_total += loss.item()

        encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        if (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, (i + 1) / num_train_samples),
                                         num_train_samples, (i + 1) / num_train_samples * 100, print_loss_avg))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  loss=losses))

    return loss.item()


def trainIters(train_loader, encoder, decoder, learning_rate=0.01):
    plot_losses = []
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # Epochs
    for epoch in range(start_epoch, epochs):
        loss = train(epoch, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
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
