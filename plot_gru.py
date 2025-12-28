import os
import re
import matplotlib.pyplot as plt

def parse_gru_log(log_path):
    epochs, val_losses, val_bleus = [], [], []

    epoch_pat = re.compile(r"Epoch:\s*(\d+)")
    valloss_pat = re.compile(r"Val\.\s*Loss:\s*([0-9.]+)")
    valbleu_pat = re.compile(r"Val\s*BLEU:\s*([0-9.]+)")

    current_epoch = None
    current_valloss = None
    current_valbleu = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = epoch_pat.search(line)
            if m:
                # 遇到新 epoch 时，先把上一组写入（如果完整）
                if current_epoch is not None and current_valloss is not None and current_valbleu is not None:
                    epochs.append(current_epoch)
                    val_losses.append(current_valloss)
                    val_bleus.append(current_valbleu)

                current_epoch = int(m.group(1))
                current_valloss = None
                current_valbleu = None
                continue

            m = valloss_pat.search(line)
            if m:
                current_valloss = float(m.group(1))
                continue

            m = valbleu_pat.search(line)
            if m:
                current_valbleu = float(m.group(1))
                continue

    # 文件结束后补上最后一个 epoch
    if current_epoch is not None and current_valloss is not None and current_valbleu is not None:
        epochs.append(current_epoch)
        val_losses.append(current_valloss)
        val_bleus.append(current_valbleu)

    return epochs, val_losses, val_bleus


def plot_gru_logs(log_files, save_dir="fig", prefix="gru"):
    os.makedirs(save_dir, exist_ok=True)

    # 1) Validation Loss
    plt.figure()
    for log_file in log_files:
        label = os.path.splitext(os.path.basename(log_file))[0]
        epochs, losses, bleus = parse_gru_log(log_file)

        if len(epochs) == 0:
            print(f"No valid epochs found in {log_file}")
            continue

        plt.plot(epochs, losses, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("GRU Ablation: Validation Loss")
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(save_dir, f"{prefix}_val_loss_curve.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Validation BLEU
    plt.figure()
    for log_file in log_files:
        label = os.path.splitext(os.path.basename(log_file))[0]
        epochs, losses, bleus = parse_gru_log(log_file)

        if len(epochs) == 0:
            continue

        plt.plot(epochs, bleus, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU")
    plt.title("GRU Ablation: Validation BLEU")
    plt.grid(True)
    plt.legend()
    bleu_path = os.path.join(save_dir, f"{prefix}_val_bleu_curve.png")
    plt.savefig(bleu_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(loss_path)
    print(bleu_path)


if __name__ == "__main__":
    log_files = [
        "./logs/gru_force_concat.log",
        "./logs/gru_force_dot.log",
        "./logs/gru_force_general.log",
        "./logs/gru_free_concat.log"
    ]
    plot_gru_logs(log_files, save_dir="fig/gru", prefix="gru")
