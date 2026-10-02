import pysam
import pandas as pd
import random
import argparse
from collections import defaultdict

def get_tes_for_chrom(te_map, chrom, start, end):
    """Filters the small TE dictionary for overlaps with a specific region."""
    return [te for te in te_map.get(chrom, []) if te[0] < end and te[1] > start]

def run_pipeline():
    parser = argparse.ArgumentParser(description="Evaluate TE effects by removal or shuffling.")
    parser.add_argument("-r", "--regions", required=True, help="BED file of 401bp regions")
    parser.add_argument("-t", "--tes", required=True, help="BED file of TE coordinates")
    parser.add_argument("-g", "--genome", required=True, help="Genome FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output FASTA path")
    parser.add_argument("-m", "--mode", choices=['remove', 'shuffle'], required=True, 
                        help="Mode: 'remove' (fills from flanks) or 'shuffle' (scrambles in-place)")
    
    args = parser.parse_args()

    genome = pysam.FastaFile(args.genome)
    
    # Load TEs into memory
    te_map = defaultdict(list)
    with open(args.tes, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                te_map[parts[0]].append((int(parts[1]), int(parts[2])))

    with open(args.output, 'w') as out_f:
        regions = pd.read_csv(args.regions, sep='\t', header=None, 
                             names=['chrom', 'start', 'end', 'name'])

        for _, reg in regions.iterrows():
            if args.mode == 'remove':
                # Fetch a wide 1200bp window to ensure we have 401bp of non-TE DNA
                midpoint = (reg.start + reg.end) // 2
                search_start, search_end = max(0, midpoint - 600), midpoint + 600
                raw_seq = genome.fetch(reg.chrom, search_start, search_end)
                chrom_tes = get_tes_for_chrom(te_map, reg.chrom, search_start, search_end)
                
                clean_chars = []
                for i in range(search_start, search_end):
                    if not any(s <= i < e for s, e in chrom_tes):
                        idx = i - search_start
                        if 0 <= idx < len(raw_seq): clean_chars.append(raw_seq[idx])
                
                clean_seq = "".join(clean_chars)
                mid = len(clean_seq) // 2
                final_seq = clean_seq[mid-200 : mid+201] # Slice 401bp
                
            else: # Shuffle mode
                raw_seq = list(genome.fetch(reg.chrom, reg.start, reg.end))
                chrom_tes = get_tes_for_chrom(te_map, reg.chrom, reg.start, reg.end)
                
                for te_s, te_e in chrom_tes:
                    idx_s, idx_e = max(0, te_s - reg.start), min(401, te_e - reg.start)
                    segment = raw_seq[idx_s:idx_e]
                    random.shuffle(segment)
                    raw_seq[idx_s:idx_e] = segment
                final_seq = "".join(raw_seq)

            if len(final_seq) == 401:
                out_f.write(f">{reg['name']}_{args.mode}\n{final_seq}\n")
            else:
                print(f"Skipping {reg['name']}: Resulting sequence length {len(final_seq)} bp")

    genome.close()

if __name__ == "__main__":
    run_pipeline()
