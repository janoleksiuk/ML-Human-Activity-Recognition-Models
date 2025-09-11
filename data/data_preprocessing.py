import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from classes import CLASS_MAPPING

"""
Example CLI usage:
python preprocessing.py \
  --input_dir data/raw/standing \
  --output_path data/processed/standing.npz \
  --class_name standing \
  --smooth moving \
  --augment
"""

def load_csv_files(input_dir):
    """Load all files from catalogue"""
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    sequences = []
    for file in all_files:
        df = pd.read_csv(file)
        X = df.values.astype(np.float32)  
        sequences.append(X)
    return sequences


def normalize(sequences):
    """z-score normalization"""
    all_data = np.vstack(sequences)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8  
    normalized = [(seq - mean) / std for seq in sequences]
    return normalized, mean, std


def create_sequences(X, seq_len=30, step=10, label="standing"):
    """sequences division"""
    sequences, labels = [], []
    for start in range(0, len(X) - seq_len + 1, step):
        end = start + seq_len
        seq = X[start:end]
        sequences.append(seq)
        labels.append(label)
    return sequences, labels


def augment_sequence(seq, noise_level=0.01):
    """data augmentation - adding gaussian noise"""
    noise = np.random.normal(0, noise_level, seq.shape)
    return seq + noise


def smooth_sequence(seq, method="sg", window_length=5, polyorder=2):
    """
    Smoothing
    
    Args:
        seq (np.ndarray): [seq_len, num_features]
        method (str): "sg" (Savitzky-Golay), "moving", "none"
        window_length (int): moving widnow filter lentgh for sg and moving filters
        polyorder (int): polymonial degree order for sg filter
    """
    if method == "none":
        return seq

    if window_length >= len(seq):
        window_length = len(seq) - 1 if len(seq) % 2 == 0 else len(seq)
    if window_length < 3:
        return seq

    if method == "sg":
        return savgol_filter(seq, window_length=window_length,
                             polyorder=polyorder, axis=0)
    elif method == "moving":
        kernel = np.ones(window_length) / window_length
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=seq)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def process_class(input_dir, output_path, class_name,
                  seq_len=30, step=5, augment=False, smooth="none"):
    """Processing for given class type"""
    if class_name not in CLASS_MAPPING:
        raise ValueError(f"Unknown class_name '{class_name}'. Must be one of {list(CLASS_MAPPING.keys())}")

    label_int = CLASS_MAPPING[class_name]

    sequences = load_csv_files(input_dir)
    sequences, mean, std = normalize(sequences)
    
    if smooth != "none":
        sequences = [smooth_sequence(seq, method=smooth) for seq in sequences]

    all_sequences, all_labels = [], []
    for seq in sequences:
        seqs, labs = create_sequences(seq, seq_len=seq_len, step=step, label=label_int)
        all_sequences.extend(seqs)
        all_labels.extend(labs)

        if augment:
            aug_seqs = [augment_sequence(s) for s in seqs]
            all_sequences.extend(aug_seqs)
            all_labels.extend(labs)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)  

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, X=X, y=y, mean=mean, std=std)

    print(f"[INFO] Saved {X.shape[0]} sequences to {output_path}")
    print(f"[INFO] Label '{class_name}' encoded as {label_int}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing CSV -> .npz")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="CSV folder")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path")
    parser.add_argument("--class_name", type=str, default="standing",
                        help="Class type (label)")
    parser.add_argument("--seq_len", type=int, default=30,
                        help="Sequence window length")
    parser.add_argument("--step", type=int, default=5,
                        help="Step of window sliding")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation?")
    parser.add_argument("--smooth", type=str, default="none",
                        choices=["none", "sg", "moving"],
                        help="Apply smoothing filter: none | sg (Savitzky-Golay) | moving (average)")

    args = parser.parse_args()

    process_class(
        input_dir=args.input_dir,
        output_path=args.output_path,
        class_name=args.class_name,
        seq_len=args.seq_len,
        step=args.step,
        augment=args.augment,
        smooth=args.smooth
    )
