import re
import os
import matplotlib.pyplot as plt

LOG_FILES = {
    "abs + layernorm": "transformer_abs_layernorm.txt",
    "abs + rmsnorm": "transformer_abs_rmsnorm.txt",
    "alibi + layernorm": "transformer_alibi_layernorm.txt",
    "alibi + rmsnorm": "transformer_alibi_rmsnorm.txt",
}

SAVE_DIR = "./fig/transformer_ablation_plots"


def parse_log(log_path):
    """
    Extract per-epoch metrics from a training log.
    Expected patterns:
      - Train Loss: X.XXX
      - Val. Loss: X.XXX
      - Val BLEU:  XX.XX
    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    train_loss = [float(x) for x in re.findall(r"Train Loss:\s*([0-9.]+)", text)]
    val_loss = [float(x) for x in re.findall(r"Val\. Loss:\s*([0-9.]+)", text)]
    val_bleu = [float(x) for x in re.findall(r"Val BLEU:\s*([0-9.]+)", text)]

    return train_loss, val_loss, val_bleu


def save_figure(fig, filename_base):
    os.makedirs(SAVE_DIR, exist_ok=True)
    png_path = os.path.join(SAVE_DIR, f"{filename_base}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")


def plot_val_loss_curves(log_dir="."):
    fig = plt.figure()

    for label, fname in LOG_FILES.items():
        path = os.path.join(log_dir, fname)
        _, val_loss, _ = parse_log(path)

        if len(val_loss) == 0:
            print(f"Warning: No validation loss found in {path}")
            continue

        epochs = list(range(1, len(val_loss) + 1))
        plt.plot(epochs, val_loss, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Transformer Ablation: Validation Loss")
    plt.legend()
    plt.grid(True)

    save_figure(fig, "transformer_val_loss_curve")
    plt.show()


def plot_val_bleu_curves(log_dir="."):
    fig = plt.figure()
    best_bleus = {}

    for label, fname in LOG_FILES.items():
        path = os.path.join(log_dir, fname)
        _, _, val_bleu = parse_log(path)

        if len(val_bleu) == 0:
            print(f"Warning: No validation BLEU found in {path}")
            best_bleus[label] = 0.0
            continue

        epochs = list(range(1, len(val_bleu) + 1))
        plt.plot(epochs, val_bleu, label=label)

        best_bleus[label] = max(val_bleu)

    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU")
    plt.title("Transformer Ablation: Validation BLEU")
    plt.legend()
    plt.grid(True)

    save_figure(fig, "transformer_val_bleu_curve")
    plt.show()

    return best_bleus


def plot_best_bleu_bar(best_bleus):
    fig = plt.figure()

    labels = list(best_bleus.keys())
    scores = [best_bleus[k] for k in labels]

    plt.bar(labels, scores)
    plt.xticks(rotation=20)
    plt.ylabel("Best Validation BLEU")
    plt.title("Transformer Ablation: Best Validation BLEU")
    plt.grid(axis="y")

    save_figure(fig, "transformer_best_bleu_bar")
    plt.show()


if __name__ == "__main__":
    log_dir = "./logs"
    plot_val_loss_curves(log_dir)
    best_bleus = plot_val_bleu_curves(log_dir)
    plot_best_bleu_bar(best_bleus)
