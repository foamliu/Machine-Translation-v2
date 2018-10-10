# import the necessary packages

import keras.backend as K

from utils import *

if __name__ == '__main__':
    input_lang = Lang('data/WORDMAP_en.json')
    output_lang = Lang('data/WORDMAP_zh.json')
    print("input_lang.n_words: " + str(input_lang.n_words))
    print("output_lang.n_words: " + str(output_lang.n_words))

    checkpoint = '{}/BEST_checkpoint.tar'.format(save_dir)  # model checkpoint
    print('checkpoint: ' + str(checkpoint))
    # Load model
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['en']
    encoder = encoder.to(device)
    encoder.eval()
    decoder = checkpoint['de']
    decoder = decoder.to(device)
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)
    for input_sentence, target_sentence in pick_n_valid_sentences(input_lang, output_lang, 10):
        decoded_words = evaluate(searcher, input_sentence, input_lang, output_lang)
        print('> {}'.format(input_sentence))
        print('= {}'.format(target_sentence))
        print('< {}'.format(''.join(decoded_words)))

    K.clear_session()
