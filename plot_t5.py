import os
import ast
import matplotlib.pyplot as plt

def parse_eval_logs(log_path):
    eval_epochs, eval_loss, eval_bleu = [], [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and "eval_loss" in line and "eval_bleu" in line:
                try:
                    d = ast.literal_eval(line)
                    eval_epochs.append(d["epoch"])
                    eval_loss.append(d["eval_loss"])
                    eval_bleu.append(d["eval_bleu"])
                except Exception:
                    continue

    return eval_epochs, eval_loss, eval_bleu


def plot_eval_curves(log_path, save_dir="fig/t5", save_prefix="t5"):
    os.makedirs(save_dir, exist_ok=True)

    epochs, losses, bleus = parse_eval_logs(log_path)
    sorted_items = sorted(zip(epochs, losses, bleus), key=lambda x: x[0])

    if len(sorted_items) == 0:
        print("No eval logs found. Make sure your log contains lines with eval_loss and eval_bleu.")
        return

    epochs, losses, bleus = zip(*sorted_items)

    # Validation Loss curve
    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("T5 Validation Loss Curve")
    plt.grid(True)
    loss_path = os.path.join(save_dir, f"{save_prefix}_val_loss_curve.png")
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Validation BLEU curve
    plt.figure()
    plt.plot(epochs, bleus, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation BLEU")
    plt.title("T5 Validation BLEU Curve")
    plt.grid(True)
    bleu_path = os.path.join(save_dir, f"{save_prefix}_val_bleu_curve.png")
    plt.savefig(bleu_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(loss_path)
    print(bleu_path)


# Usage
plot_eval_curves("./logs/t5_log.txt", save_dir="fig/t5", save_prefix="t5")
