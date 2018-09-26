import math
import random
import time

import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch import optim

from data_generator import TranslationDataset
from models import EncoderRNN, AttnDecoderRNN
from utils import *

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
    encoder_hidden = encoder.init_hidden()

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
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    print_loss_total = 0  # Reset every print_every

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)

    start = time.time()

    # Batches
    for i, (input_array, target_array) in enumerate(train_loader):
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
        losses.update(loss.item(), target_length)
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_every == 0:
            print_loss_total = 0
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))

    return loss.item()


def validate(val_loader, encoder, decoder, criterion):
    encoder.eval()  # eval mode (no dropout or batchnorm)
    decoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (input_array, target_array) in enumerate(val_loader):
        # Move to GPU, if available
        input_tensor = torch.tensor(input_array, device=device).view(-1, 1)
        target_tensor = torch.tensor(target_array, device=device).view(-1, 1)
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        loss = calc_loss(input_tensor, input_length, target_tensor, target_length, encoder, decoder, criterion)

        # Keep track of metrics
        losses.update(loss.item(), target_length)
        batch_time.update(time.time() - start)

        start = time.time()

        print('Validation: [{0}/{1}]\t'
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                              loss=losses))

        # Store references (true captions), and hypothesis (prediction) for each pair
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        reference = [output_lang.index2word[idx.item()] for idx in target_array]
        references.append([reference])

        # Hypotheses
        sentence_en = ' '.join([input_lang.index2word[idx.item()] for idx in input_array])
        print('sentence_en: ' + str(sentence_en))
        preds, _ = evaluate(encoder, decoder, input_tensor, input_length)
        hypotheses.append(preds)

        print('preds: ' + str(preds))
        print('reference: ' + str(reference))

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            bleu=bleu4))

    return bleu4


def main():
    word_map_zh = json.load(open('data/WORDMAP_zh.json', 'r'))
    word_map_en = json.load(open('data/WORDMAP_en.json', 'r'))

    input_lang_n_words = len(word_map_en)
    output_lang_n_words = len(word_map_zh)

    train_loader = torch.utils.data.DataLoader(
        TranslationDataset('train'), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        TranslationDataset('valid'), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    encoder = EncoderRNN(input_lang_n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang_n_words, dropout_p=dropout_p).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    plot_losses = []

    # Epochs
    for epoch in range(start_epoch, epochs):
        # # One epoch's training
        # loss = train(epoch, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        # plot_losses.append(loss)

        # One epoch's validation
        validate(val_loader=val_loader,
                 encoder=encoder,
                 decoder=decoder,
                 criterion=criterion)

    showPlot(plot_losses)


if __name__ == '__main__':
    main()
