import sentencepiece as spm

# Chinese BPE
spm.SentencePieceTrainer.train(
    input="data/train_100k.zh.txt",
    model_prefix="spm_zh",
    vocab_size=16000,
    model_type="bpe",
    character_coverage=0.9995,
    unk_id=0,
    pad_id=1,
    bos_id=2,
    eos_id=3
)

# English BPE
spm.SentencePieceTrainer.train(
    input="data/train_100k.en.txt",
    model_prefix="spm_en",
    vocab_size=16000,
    model_type="bpe",
    character_coverage=1.0,
    unk_id=0,
    pad_id=1,
    bos_id=2,
    eos_id=3
)
