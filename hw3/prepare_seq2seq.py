# coding: utf-8

# ## Prepare parallel corpus
# 
# **Based on TensorFlow code: https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py**

# In[ ]:

import os
import re
import pickle
from tqdm import tqdm
import sys


# In[ ]:

from nmt_config import *


# In[ ]:

data_fname = {"en": os.path.join(data_dir, "text_all.en"),
              "fr": os.path.join(data_dir, "text_all.fr")}


# In[ ]:

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':~;)(])")
_DIGIT_RE = re.compile(br"\d")


# In[ ]:

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.sub(b"", w) for w in _WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() for w in words if w]


# In[ ]:

def extract_k_lines(fr_fname, en_fname, k):
    num_lines = 0
    with open(data_fname["fr"],"rb") as f_fr, open(data_fname["en"],"rb") as f_en:
        with open(fr_fname,"wb") as out_fr, open(en_fname,"wb") as out_en:
            for i, (line_fr, line_en) in enumerate(zip(f_fr, f_en)):
                if num_lines >= k:
                    break
                words_fr = basic_tokenizer(line_fr)
                words_en = basic_tokenizer(line_en)
                if len(words_fr) > 0 and len(words_en) > 0:
                    # write to tokens file
                    out_fr.write(b" ".join(words_fr) + b"\n")
                    out_en.write(b" ".join(words_en) + b"\n")
                    num_lines += 1
        print("Total lines={0:d}, valid lines={1:d}".format(i, num_lines))            
        print("finished writing {0:s} and {1:s}".format(fr_fname, en_fname))
    


# In[ ]:

def create_vocab(text_fname, num_train, max_vocabulary_size, freq_thresh):
    vocab = {}
    w2i = {}
    i2w = {}
    with open(text_fname,"rb") as in_f:
        for i, line in enumerate(in_f):
            if i >= num_train:
                break
            
            words = line.strip().split()
            for w in words:
                word = _DIGIT_RE.sub(b"0", w)
                word = _WORD_SPLIT.sub(b"", w)
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    
    print("vocab length before: {0:d}".format(len(vocab)))
    vocab = {k:vocab[k] for k in vocab if vocab[k] > freq_thresh}
    print("vocab length after: {0:d}".format(len(vocab)))
    
    vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    print("Finished generating vocabulary")
    if len(vocab_list) > max_vocabulary_size:
        print("Vocab size={0:d}, trimmed to max={1:d}".format(len(vocab_list), max_vocabulary_size))
        vocab_list = vocab_list[:max_vocabulary_size]
    else:
        print("Vocab size={0:d}".format(len(vocab_list)))

    for i, w in enumerate(vocab_list):
        w2i[w] = i
        i2w[i] = w
            
    print("finished vocab processing for {0:s}".format(text_fname))
    
    for k in vocab:
        if vocab[k] <= freq_thresh:
            print("Ahaaaaa!!!", k, vocab[k])
    
    return vocab, w2i, i2w


# In[ ]:

def create_input_config(k, num_train=NUM_TRAINING_SENTENCES, freq_thresh=FREQ_THRESH):
    # Output file names
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        
    en_name = os.path.join(input_dir, "text.en")
    fr_name = os.path.join(input_dir, "text.fr")
    
    en_tokens_name = os.path.join(input_dir, "tokens.en")
    fr_tokens_name = os.path.join(input_dir, "tokens.fr")
    
    vocab_path = os.path.join(input_dir, "vocab.dict")
    w2i_path = os.path.join(input_dir, "w2i.dict")
    i2w_path = os.path.join(input_dir, "i2w.dict")
    
    # extract k lines
    extract_k_lines(fr_name, en_name, k)
    
    # create vocabularies
    vocab = {"en":{}, "fr":{}}
    w2i = {"en":{}, "fr":{}}
    i2w = {"en":{}, "fr":{}}
    
    print("*"*50)
    print("en file")
    print("*"*50)
    vocab["en"], w2i["en"], i2w["en"] = create_vocab(en_name, 
                                                     num_train=NUM_TRAINING_SENTENCES,
                                                     max_vocabulary_size=max_vocab_size["en"], 
                                                     freq_thresh=FREQ_THRESH)
    print("*"*50)
    print("fr file")
    print("*"*50)
    vocab["fr"], w2i["fr"], i2w["fr"] = create_vocab(fr_name, 
                                                     num_train=NUM_TRAINING_SENTENCES,
                                                     max_vocabulary_size=max_vocab_size["fr"], 
                                                     freq_thresh=FREQ_THRESH)
    print("*"*50)
    
    pickle.dump(vocab, open(vocab_path, "wb"))
    pickle.dump(w2i, open(w2i_path, "wb"))
    pickle.dump(i2w, open(i2w_path, "wb"))
    print("finished creating input config for {0:d} lines".format(k))

# In[ ]:

create_input_config(k=NUM_SENTENCES, num_train=NUM_TRAINING_SENTENCES, freq_thresh=FREQ_THRESH)


# In[ ]:

