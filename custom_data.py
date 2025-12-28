import torch
import json
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import sentencepiece as spm

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

class LanguageProcessor:
    """
    SentencePiece-based language processor.
    Replaces word-level tokenization and manual vocabulary.
    """

    def __init__(self, model_path, max_len=80):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        self.max_len = max_len

        # Must match SentencePiece training
        self.UNK_IDX = 0
        self.PAD_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3


    def __len__(self):
        return self.sp.get_piece_size()

    def encode(self, sentence):
        """
        Encode sentence into subword ids with <sos> and <eos>.
        """
        ids = self.sp.encode(sentence, out_type=int)
        ids = ids[: self.max_len - 2]
        return [self.SOS_IDX] + ids + [self.EOS_IDX]

    def decode(self, ids):
        """
        Decode subword ids back into string.
        """
        ids = [i for i in ids if i not in (
            self.PAD_IDX, self.SOS_IDX, self.EOS_IDX
        )]
        return self.sp.decode(ids)


# --- 2. NMTDataset: Simplified to use Processors ---

class NMTDataset(Dataset):
    """
    NMT Dataset using SentencePiece processors.
    JSONL format:
    {
        "zh": "...",
        "en": "..."
    }
    """

    def __init__(self, data_path, src_processor, tgt_processor):
        self.data = []
        self.src_processor = src_processor
        self.tgt_processor = tgt_processor


        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.data.append((obj["zh"], obj["en"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        zh_sent, en_sent = self.data[idx]

        # SentencePiece encoding (subword-level)
        src_ids = self.src_processor.encode(zh_sent)
        tgt_ids = self.tgt_processor.encode(en_sent)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )

# --- 3. Collate Function ---

def collate_fn(batch, pad_idx, batch_first=False):
    """
    Pads sequences in a batch.
    """
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(
        src_batch, padding_value=pad_idx, batch_first=batch_first
    )
    tgt_batch = pad_sequence(
        tgt_batch, padding_value=pad_idx, batch_first=batch_first
    )

    return src_batch, tgt_batch

# --- 4. Main Loading Function (Integration Logic) ---

def load_data_and_get_loaders(
    train_path,
    valid_path,
    test_path,
    batch_size=128,
    batch_first=False,
):
    """
    Load data using SentencePiece-based processors.
    """

    print("Initializing SentencePiece processors...")

    # Initialize processors with pretrained SentencePiece models
    src_processor = LanguageProcessor(
        model_path="./spm/spm_zh.model",
        max_len=200
    )

    tgt_processor = LanguageProcessor(
        model_path="./spm/spm_en.model",
        max_len=200
    )

    PAD_IDX = src_processor.PAD_IDX  # must be 1

    print("Creating datasets...")
    train_dataset = NMTDataset(train_path, src_processor, tgt_processor)
    valid_dataset = NMTDataset(valid_path, src_processor, tgt_processor)
    test_dataset  = NMTDataset(test_path,  src_processor, tgt_processor)

    print(
        f"Dataset sizes - "
        f"Train: {len(train_dataset)}, "
        f"Valid: {len(valid_dataset)}, "
        f"Test: {len(test_dataset)}"
    )

    collate_func = partial(
        collate_fn,
        pad_idx=PAD_IDX,
        batch_first=batch_first
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_func,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_func,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_func,
    )

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
        src_processor,
        tgt_processor,
    )