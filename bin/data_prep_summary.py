import os
import argparse
import pandas as pd
import csv
import json
import struct
import wave
from pathlib import Path
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_output_dir", type=str, required=True, help="Path to the directory where the processed answers data will be saved.")
    args = parser.parse_args()

    metadata = pd.read_csv(Path(args.answers_output_dir) / "metadata.csv")
    metadata['question_duration'] = metadata['question_end_time'] - metadata['context_end_time']
    metadata['context_duration'] = metadata['context_end_time']
    print(f"for data in {args.answers_output_dir}:\
          \n\t{len(metadata)} total examples\
          \n\tContext durations (s):\tmin={metadata['context_duration'].min():.2f}, max={metadata['context_duration'].max():.2f}, mean={metadata['context_duration'].mean():.2f}\
          \n\tQuestion durations (s):\tmin={metadata['question_duration'].min():.2f}, max={metadata['question_duration'].max():.2f}, mean={metadata['question_duration'].mean():.2f}\
          \n\tAnswer durations (s):\tmin={metadata['answer_duration'].min():.2f}, max={metadata['answer_duration'].max():.2f}, mean={metadata['answer_duration'].mean():.2f}")


