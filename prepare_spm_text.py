import json

def extract_text(jsonl_path, zh_out, en_out):
    with open(jsonl_path, "r", encoding="utf-8") as f, \
         open(zh_out, "w", encoding="utf-8") as f_zh, \
         open(en_out, "w", encoding="utf-8") as f_en:
        for line in f:
            obj = json.loads(line)
            f_zh.write(obj["zh"].strip() + "\n")
            f_en.write(obj["en"].strip() + "\n")


if __name__ == "__main__":
    # use the SAME training set you actually train on
    extract_text(
        jsonl_path="data/train_100k.jsonl",
        zh_out="data/train_100k.zh.txt",
        en_out="data/train_100k.en.txt"
    )
