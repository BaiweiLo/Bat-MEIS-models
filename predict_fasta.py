import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from Bio import SeqIO

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Initialize matrix of (len, 4)
    encoding = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            encoding[i, mapping[base]] = 1.0
    return encoding

def run_prediction(model_path, fasta_path, input_length):
    # 1. Load the model
    try:
        model = load_model(model_path)
        print(f"Successfully loaded model: {model_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return

    predictions = []
    
    # 2. Iterate through FASTA
    # We stream sequences to keep memory usage low
    with open(fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)

            # 3. Check for length discrepancy
            if len(sequence) != input_length:
                print(f"ERROR: Sequence '{seq_id}' has length {len(sequence)}, "
                      f"but expected {input_length}.", file=sys.stderr)
                sys.exit(1)

            # 4. Process sequence
            encoded_seq = one_hot_encode(sequence)
            # Reshape for model input: (1, length, 4)
            X = np.expand_dims(encoded_seq, axis=0)
            
            # 5. Predict
            pred_val = model.predict(X, verbose=0)
            
            # If multi-label/output, flatten or join the values
            if pred_val.shape[1] > 1:
                pred_str = "\t".join([f"{v:.6f}" for v in pred_val[0]])
            else:
                pred_str = f"{pred_val[0][0]:.6f}"
                
            predictions.append([seq_id, pred_str])

    # 6. Output to TSV
    df = pd.DataFrame(predictions)
    df.to_csv(sys.stdout, sep='\t', index=False, header=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict_fasta.py <model.h5> <input.fasta> <length>")
    else:
        m_path = sys.argv[1]
        f_path = sys.argv[2]
        i_len = int(sys.argv[3])
        run_prediction(m_path, f_path, i_len)
