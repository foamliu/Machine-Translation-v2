import math
import random
import time

import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch import optim

from data_gen import TranslationDataset
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


def calc_loss(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder):
    # Initialize variables
    loss = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss

    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss

    return loss, mask_loss, nTotal


def train(epoch, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    print_loss_total = 0  # Reset every print_every

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)

    start = time.time()

    # Batches
    for i in range(train_loader.__len__()):
        input_variable, lengths, target_variable, mask, max_target_len = train_loader.__getitem__(i)

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

    train_loader = TranslationDataset('train')
    val_loader = TranslationDataset('valid')

    encoder = EncoderRNN(input_lang_n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang_n_words, dropout_p=dropout_p).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    plot_losses = []

    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        loss = train(epoch, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        plot_losses.append(loss)

        # One epoch's validation
        validate(val_loader=val_loader,
                 encoder=encoder,
                 decoder=decoder,
                 criterion=criterion)

    showPlot(plot_losses)


if __name__ == '__main__':
    main()
