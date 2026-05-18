import csv
import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import spacy
from tqdm import tqdm

# ======================
# CONFIG
# ======================
ALL_JSONS_TXT = Path("/home/ahallur1/spear/Seamless_Experiments/Lexical/all_jsons.txt")
OUT_DIR = Path("/home/ahallur1/spear/Seamless_Experiments/Lexical/shard_csvs")

# Lexical diversity parameters (literature-backed)
MATTR_SMALL = 50
MATTR_LARGE = 500
MTLD_TTR_THRESHOLD = 0.72

# Minimums for meaningful LD
MIN_WORDS_LD = 50

# ASR confidence diagnostics
LOW_CONF_THRESH = 0.7

# ======================
# CSV HEADER
# ======================
CSV_HEADER = [
    # identifiers
    "orig_id",
    "wav_path",

    # status
    "lexical_status",
    "status_reason",

    # size & confidence
    "total_words",
    "unique_words",
    "mean_asr_confidence",
    "low_conf_flag",

    # lexical density
    "content_word_count",
    "function_word_count",
    "lexical_density",

    # diversity
    "ttr",
    "mattr_small",
    "mattr_large",
    "mattr_ratio",
    "mtld",

    # distribution shape
    "hapax_ratio",
    "lexical_entropy",

    # discourse
    "backchannel_ratio",
    "discourse_marker_ratio",
]

# ======================
# HELPERS
# ======================
def normalize_word(w):
    return w.lower().strip(".,!?;:\"()[]{}")


def compute_entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return np.nan
    probs = np.array(list(counter.values())) / total
    return -np.sum(probs * np.log2(probs))


def compute_mattr(tokens, window):
    if len(tokens) < window:
        return np.nan
    ttrs = []
    for i in range(len(tokens) - window + 1):
        window_tokens = tokens[i:i+window]
        ttrs.append(len(set(window_tokens)) / window)
    return float(np.mean(ttrs))


def compute_mtld(tokens, threshold=0.72):
    def mtld_pass(seq):
        factors = 0
        types = set()
        token_count = 0

        for tok in seq:
            token_count += 1
            types.add(tok)
            ttr = len(types) / token_count

            if ttr <= threshold:
                factors += 1
                types = set()
                token_count = 0

        if token_count > 0:
            factors += (1 - ttr) / (1 - threshold)

        return len(seq) / factors if factors > 0 else np.nan

    if len(tokens) < MIN_WORDS_LD:
        return np.nan

    forward = mtld_pass(tokens)
    backward = mtld_pass(tokens[::-1])

    return float(np.nanmean([forward, backward]))


# ======================
# MAIN LOOP
# ======================
def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python run_lexical_extraction.py <shard_idx> <num_shards>")
        return 1

    shard_idx = int(sys.argv[1])
    num_shards = int(sys.argv[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"lexical_shard_{shard_idx:04d}.csv"
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    with open(ALL_JSONS_TXT) as f:
        all_jsons = [line.strip() for line in f if line.strip()]

    shard_jsons = all_jsons[shard_idx :: num_shards]
    print(f"Shard {shard_idx}: processing {len(shard_jsons)} JSON files")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()

        for json_path_str in tqdm(shard_jsons):
            json_path = Path(json_path_str)

            row = {
                "orig_id": json_path.stem,
                "wav_path": "",
                "lexical_status": "OK",
                "status_reason": "",
            }

            try:
                with open(json_path) as jf:
                    data = json.load(jf)

                row["wav_path"] = data.get("id", "")

                sentences = data.get("metadata:transcript", [])

                tokens = []
                confidences = []
                sentence_texts = []

                for sent in sentences:
                    words = sent.get("words", [])
                    sent_tokens = []

                    for w in words:
                        word = normalize_word(w.get("word", ""))
                        if not word:
                            continue
                        tokens.append(word)
                        sent_tokens.append(word)
                        confidences.append(w.get("score", np.nan))

                    if sent_tokens:
                        sentence_texts.append(" ".join(sent_tokens))

                total_words = len(tokens)

                if total_words == 0:
                    row["lexical_status"] = "EMPTY"
                    writer.writerow(row)
                    continue

                row["total_words"] = total_words
                row["unique_words"] = len(set(tokens))

                mean_conf = float(np.nanmean(confidences))
                row["mean_asr_confidence"] = mean_conf
                row["low_conf_flag"] = mean_conf < LOW_CONF_THRESH

                # POS tagging
                doc = nlp(" ".join(tokens))
                pos_tags = [t.pos_ for t in doc]

                content_mask = [p in {"NOUN", "VERB", "ADJ", "ADV"} for p in pos_tags]
                content_count = sum(content_mask)
                function_count = total_words - content_count

                row["content_word_count"] = content_count
                row["function_word_count"] = function_count
                row["lexical_density"] = content_count / total_words

                # Diversity
                row["ttr"] = row["unique_words"] / total_words

                if total_words >= MIN_WORDS_LD:
                    row["mattr_small"] = compute_mattr(tokens, MATTR_SMALL)
                    row["mattr_large"] = compute_mattr(
                        tokens, min(MATTR_LARGE, total_words)
                    )
                    if row["mattr_small"] and row["mattr_large"]:
                        row["mattr_ratio"] = row["mattr_small"] / row["mattr_large"]
                    else:
                        row["mattr_ratio"] = np.nan

                    row["mtld"] = compute_mtld(tokens)
                else:
                    row["mattr_small"] = np.nan
                    row["mattr_large"] = np.nan
                    row["mattr_ratio"] = np.nan
                    row["mtld"] = np.nan
                    row["status_reason"] = "SHORT_TEXT"

                # Distribution shape
                freq = Counter(tokens)
                row["hapax_ratio"] = sum(1 for c in freq.values() if c == 1) / total_words
                row["lexical_entropy"] = compute_entropy(freq)

                # Discourse & backchannels (sentence-level)
                backchannels = 0
                discourse = 0

                for sent in sentence_texts:
                    words = sent.split()
                    if len(words) <= 2:
                        if any(w in {"yeah", "okay", "ok", "uh", "um"} for w in words):
                            backchannels += 1
                    else:
                        if any(
                            phrase in sent
                            for phrase in ["you know", "i mean", "kind of", "sort of"]
                        ):
                            discourse += 1

                row["backchannel_ratio"] = backchannels / len(sentence_texts) if sentence_texts else 0
                row["discourse_marker_ratio"] = discourse / len(sentence_texts) if sentence_texts else 0

                writer.writerow(row)

            except Exception as e:
                row["lexical_status"] = "ERROR"
                row["status_reason"] = str(e)
                writer.writerow(row)
                traceback.print_exc()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
