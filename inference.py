import os
import sys
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import get_args
from custom_data import load_data_and_get_loaders, PAD_IDX
from translate import translate_sentence_rnn

from models.gru import Encoder as GRUEncoder, Decoder as GRUDecoder, Seq2Seq as GRUSeq2Seq, Attention
from models.transformer import Encoder as TransEncoder, Decoder as TransDecoder, Transformer


# ============================================================
# 0) Predefined model paths (YOU EDIT THESE ONCE)
# ============================================================
MODEL_PATHS = {
    "gru": "ckpts/gru/force/concat/best_model.pt",
    "transformer": "ckpts/transformer/absolute/rmsnorm/best_model.pt",
    "t5": "ckpts/t5_best",   # directory OR huggingface model name
}

# ============================================================
# 1) Transformer Inference Wrapper
# ============================================================
@torch.no_grad()
def translate_sentence_transformer(sentence, src_vocab, trg_vocab, model, device, max_len=200):
    model.eval()
    BOS_IDX, EOS_IDX = 2, 3

    src_indexes = src_vocab.encode(sentence)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [BOS_IDX]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break

    return trg_vocab.decode(trg_indexes)


# ============================================================
# 2) T5 Translation Logic
# ============================================================
@torch.no_grad()
def translate_sentence_t5(sentence, tokenizer, model, device, max_len=200):
    model.eval()
    prefix = "translate Chinese to English: "
    input_text = prefix + sentence

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=max_len, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# 3) Load processors once
# ============================================================
def load_processors(args, model_name):
    train_path = os.path.join(args.data_dir, f'train_{args.dataset_size}.jsonl')
    valid_path = os.path.join(args.data_dir, 'valid.jsonl')
    test_path  = os.path.join(args.data_dir, 'test.jsonl')

    batch_first = True if model_name == 'transformer' else False

    *_, src_processor, tgt_processor = load_data_and_get_loaders(
        train_path, valid_path, test_path,
        batch_size=1, batch_first=batch_first
    )
    return src_processor, tgt_processor


# ============================================================
#  4) Determine checkpoint path: use predefined MODEL_PATHS
# ============================================================
def get_ckpt_path(model_name, override=None):
    # 1) user override has highest priority
    if override is not None:
        return override

    # 2) use predefined
    if model_name in MODEL_PATHS:
        return MODEL_PATHS[model_name]

    raise ValueError(f"Model '{model_name}' has no predefined model_path!")


# ============================================================
# 5) Load model by name
# ============================================================
def load_model(args, model_name, device, src_processor=None, tgt_processor=None, ckpt_override=None):
    ckpt_path = get_ckpt_path(model_name, ckpt_override)
    print(f"[Load] Model={model_name}, ckpt={ckpt_path}")

    if model_name in ["gru", "transformer"]:
        input_dim = len(src_processor)
        output_dim = len(tgt_processor)

    # -------- GRU --------
    if model_name == "gru":
        attn = Attention(args.hid_dim, args.hid_dim, method=args.rnn_attn_type)
        enc = GRUEncoder(input_dim, args.emb_dim, args.hid_dim, args.hid_dim, 0.5, n_layers=2)
        dec = GRUDecoder(output_dim, args.emb_dim, args.hid_dim, args.hid_dim, 0.5, attn, n_layers=2)
        model = GRUSeq2Seq(enc, dec, device).to(device)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"GRU checkpoint not found: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model, None

    # -------- Transformer --------
    elif model_name == "transformer":
        enc = TransEncoder(input_dim, args.hid_dim, args.n_layers, args.n_heads, 
                           args.pf_dim, args.dropout, device, PAD_IDX, norm_type=args.norm_type, pos_type=args.pos_type)
        dec = TransDecoder(output_dim, args.hid_dim, args.n_layers, args.n_heads, 
                           args.pf_dim, args.dropout, device, PAD_IDX, norm_type=args.norm_type, pos_type=args.pos_type)
        model = Transformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model, None

    # -------- T5 --------
    elif model_name == "t5":
        model = T5ForConditionalGeneration.from_pretrained(ckpt_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(ckpt_path)
        return model, tokenizer

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
#  6) Interactive loop with runtime model switching
# ============================================================
def interactive_loop(args, device):
    print("\n" + "="*75)
    print(" Interactive Translation (Runtime Switch + Predefined model paths)")
    print(" Commands:")
    print("   /model gru | transformer | t5      -> switch model (auto load predefined ckpt)")
    print("   /ckpt  PATH                       -> override ckpt path for NEXT model load")
    print("   /reset                            -> reset ckpt override and use predefined")
    print("   /show                             -> show current model & ckpt")
    print("   /help                             -> show commands")
    print("   /exit                             -> quit")
    print("="*75 + "\n")

    processors_cache = {}
    current_model_name = args.model
    current_ckpt_override = None  # override only, not default
    current_model = None
    current_tokenizer = None
    src_proc, tgt_proc = None, None

    def ensure_loaded(model_name):
        nonlocal current_model, current_tokenizer, src_proc, tgt_proc, current_ckpt_override

        if model_name in ["gru", "transformer"]:
            if model_name not in processors_cache:
                processors_cache[model_name] = load_processors(args, model_name)
            src_proc, tgt_proc = processors_cache[model_name]

        current_model, current_tokenizer = load_model(
            args=args,
            model_name=model_name,
            device=device,
            src_processor=src_proc,
            tgt_processor=tgt_proc,
            ckpt_override=current_ckpt_override
        )
        print(f"[Ready] Current model = {model_name.upper()}")

    # initial load
    try:
        ensure_loaded(current_model_name)
    except Exception as e:
        print(f"[Init Load Error] {e}")
        return

    while True:
        user_input = input("Chinese Input (or /cmd): ").strip()

        if not user_input:
            continue

        # ---------- Commands ----------
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ["/exit", "/quit"]:
                break

            elif cmd == "/help":
                print("Commands:")
                print("  /model gru|transformer|t5     Switch model (uses predefined ckpt)")
                print("  /ckpt path/to/xxx.pt          Override checkpoint (only next load)")
                print("  /reset                        Reset override -> back to predefined")
                print("  /show                         Show current model")
                print("  /exit                         Quit")
                continue

            elif cmd == "/show":
                print(f"[Info] current_model={current_model_name}")
                print(f"[Info] predefined_ckpt={MODEL_PATHS[current_model_name]}")
                print(f"[Info] override_ckpt={current_ckpt_override}")
                continue

            elif cmd == "/reset":
                current_ckpt_override = None
                print("[Info] override cleared, will use predefined ckpt next time.")
                continue

            elif cmd == "/ckpt":
                if len(parts) < 2:
                    print("Usage: /ckpt path/to/checkpoint_or_dir")
                    continue
                current_ckpt_override = parts[1]
                print(f"[Info] override ckpt set: {current_ckpt_override}")
                print("Now reload model by running: /model <name>")
                continue

            elif cmd == "/model":
                if len(parts) < 2:
                    print("Usage: /model gru|transformer|t5")
                    continue
                new_model = parts[1].lower()
                if new_model not in ["gru", "transformer", "t5"]:
                    print("Invalid model. Choose from: gru, transformer, t5")
                    continue

                current_model_name = new_model
                try:
                    ensure_loaded(current_model_name)
                except Exception as e:
                    print(f"[Switch Error] {e}")
                continue

            else:
                print("Unknown command. Type /help for commands.")
                continue

        # ---------- Translation ----------
        try:
            if current_model_name == "gru":
                result = translate_sentence_rnn(user_input, src_proc, tgt_proc, current_model, device, max_len=200)
            elif current_model_name == "transformer":
                result = translate_sentence_transformer(user_input, src_proc, tgt_proc, current_model, device, max_len=200)
            else:
                result = translate_sentence_t5(user_input, current_tokenizer, current_model, device, max_len=200)

            print(f"English Output: \033[92m{result}\033[0m\n")

        except Exception as e:
            print(f"[Translation Error] {e}\n")

    print("Exiting...")


# ============================================================
# main
# ============================================================
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on device: {device}")

    print("\n[Predefined Model Paths]")
    for k, v in MODEL_PATHS.items():
        print(f"  {k:12s}: {v}")
    print()

    interactive_loop(args, device)


if __name__ == "__main__":
    main()
