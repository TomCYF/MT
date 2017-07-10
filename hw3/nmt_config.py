import os

#---------------------------------------------------------------------
# Data Parameters
#---------------------------------------------------------------------
max_vocab_size = {"en" : 200000, "fr" : 200000}

# Special vocabulary symbols - we always put them at the start.
PAD = b"_PAD"
GO = b"_GO"
EOS = b"_EOS"
UNK = b"_UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

NO_ATTN = 0
SOFT_ATTN = 1

# for appending post fix to output
attn_post = ["NO_ATTN", "SOFT_ATTN"]

NUM_SENTENCES = 50000

DATASET = ["OPEN_SUB", "INUKTITUT"][1]

EXP_NAME_PREFIX="baseline"

if DATASET == "OPEN_SUB":
#-----------------------------------------------------------------
# Open subtitles configuration
#-----------------------------------------------------------------
    print("Open subtitles dataset configuration")
    # subtitles data
    model_dir = os.path.join("hu_en_model_{0:d}".format(NUM_SENTENCES))
    input_dir = os.path.join("hu_en_data_{0:d}".format(NUM_SENTENCES))
    data_dir = os.path.join("hu_en_data")
    # Subtitiles data
    # use 90% of the data for training
    NUM_TRAINING_SENTENCES = (NUM_SENTENCES * 90) // 100
    # remaining (max 10%) left to be used for dev. For training, we limit the dev size to 500 to speed up perplexity and Bleu computation
    NUM_DEV_SENTENCES = 500
    FREQ_THRESH = 0
    BATCH_SIZE = 64
    # A total of 7 buckets, with a length range of 3 each, giving total
    # BUCKET_WIDTH * NUM_BUCKETS = 21 for e.g.
    BUCKET_WIDTH = 3
    NUM_BUCKETS = 7
    MAX_PREDICT_LEN = BUCKET_WIDTH*NUM_BUCKETS
    if NUM_SENTENCES == 50000:
        # 50K
        EXP_NAME= EXP_NAME_PREFIX + "_budoslab"
    elif NUM_SENTENCES == 100000:
        # 100K
        EXP_NAME= EXP_NAME_PREFIX + "_bogomasina"
#-----------------------------------------------------------------
elif DATASET == "INUKTITUT":
#-----------------------------------------------------------------
# Inuktitut English configuration
#-----------------------------------------------------------------
    print("Inuktitut English dataset configuration")
    model_dir = os.path.join("in_en_model_{0:d}".format(NUM_SENTENCES))
    input_dir = os.path.join("in_en_data_{0:d}".format(NUM_SENTENCES))
    data_dir = os.path.join("in_en_data")
    # use 90% of the data for training
    NUM_TRAINING_SENTENCES = (NUM_SENTENCES * 90) // 100
    # remaining (max 10%) left to be used for dev. For training, we limit the dev size to 500 to speed up perplexity and Bleu computation
    NUM_DEV_SENTENCES = 500
    FREQ_THRESH = 0
    BATCH_SIZE = 64
    # A total of 10 buckets, with a length range of 3 each, giving total
    # BUCKET_WIDTH * NUM_BUCKETS = 30 for e.g.
    BUCKET_WIDTH = 3
    NUM_BUCKETS = 10
    MAX_PREDICT_LEN = BUCKET_WIDTH*NUM_BUCKETS
    if NUM_SENTENCES == 50000:
        # 50K
        EXP_NAME= EXP_NAME_PREFIX + "_aajuq"
    elif NUM_SENTENCES == 100000:
        # 100K
        EXP_NAME= EXP_NAME_PREFIX + "_ailliijuq"
#-----------------------------------------------------------------

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(input_dir):
    print("Input folder not found".format(input_dir))

text_fname = {"en": os.path.join(input_dir, "text.en"), "fr": os.path.join(input_dir, "text.fr")}
bucket_data_fname = os.path.join(input_dir, "buckets_{0:d}.list")
tokens_fname = os.path.join(input_dir, "tokens.list")
vocab_path = os.path.join(input_dir, "vocab.dict")
w2i_path = os.path.join(input_dir, "w2i.dict")
i2w_path = os.path.join(input_dir, "i2w.dict")
#---------------------------------------------------------------------
# Model Parameters
#---------------------------------------------------------------------
num_layers_enc = 3
num_layers_dec = 3
use_attn = SOFT_ATTN
#---------------------------------------------------------------------
# !! NOTE !!
#---------------------------------------------------------------------
# FOR INUKTITUT-ENGLISH baseline model, the hidden units should be set to 200
# FOR HUNGARIAN-ENGLISH baseline model, the hidden units should be set to 100
hidden_units = 200

load_existing_model = True
create_buckets_flag = False
#---------------------------------------------------------------------
# Training Parameters
#---------------------------------------------------------------------

#---------------------------------------------------------------------
# Training EPOCHS
#---------------------------------------------------------------------
# if 0 - will only load a previously saved model if it exists
#---------------------------------------------------------------------
NUM_EPOCHS = 0

# Change the dev set to include all the sentences not used for training, instead of 500
# Using all during training impacts timing
if NUM_EPOCHS == 0:
    NUM_DEV_SENTENCES = NUM_SENTENCES-NUM_TRAINING_SENTENCES

#---------------------------------------------------------------------
# GPU/CPU
#---------------------------------------------------------------------
# if >= 0, use GPU, if negative use CPU
gpuid = -1
#---------------------------------------------------------------------
# Log file details
#---------------------------------------------------------------------
name_to_log = "{0:d}sen_{1:d}-{2:d}layers_{3:d}units_{4:s}_{5:s}".format(
                                                            NUM_SENTENCES,
                                                            num_layers_enc,
                                                            num_layers_dec,
                                                            hidden_units,
                                                            EXP_NAME,
                                                            attn_post[use_attn])

log_train_fil_name = os.path.join(model_dir, "train_{0:s}.log".format(name_to_log))
log_dev_fil_name = os.path.join(model_dir, "dev_{0:s}.log".format(name_to_log))
model_fil = os.path.join(model_dir, "seq2seq_{0:s}.model".format(name_to_log))
#---------------------------------------------------------------------
