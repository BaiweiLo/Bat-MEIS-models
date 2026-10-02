from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('/project/bat_analysis/baiwei/software/Neural_Network_DNA_Demo/') 
from helper import IOHelper, SequenceHelper # from https://github.com/const-ae/Neural_Network_DNA_Demo 
### Important!! One hot encoding was manually changed to A, C, G, T in order to match TF-modisco
import random
random.seed(1234)

# load the accessibility model
main_model = load_model('model.h5')

# load the peaks projected from mouse to bats (DC- in IPP), +-250bp extended
path='.'
input_fasta_data = IOHelper.get_fastas_from_file(path + '/' +  'DC-_501.fasta', uppercase=True)
seq_to_eval = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 501, SequenceHelper.parse_alpha_to_seq)

# Define functions used for saturation mutagenesis
nucleotide_map = np.array([
    [1, 0, 0, 0],  # A
    [0, 1, 0, 0],  # C
    [0, 0, 1, 0],  # G
    [0, 0, 0, 1],  # T
])

# Reverse mapping for decoding
reverse_map = {tuple(v): k for k, v in zip('ACGT', nucleotide_map)}

def one_hot_to_sequence(one_hot_seq):
    """Converts a one-hot encoded sequence back to a nucleotide sequence."""
    return ''.join(reverse_map[tuple(base)] for base in one_hot_seq)

def mutate_sequence(sequence, position, new_base):
    """Mutates the sequence at a specific position to a new base."""
    mutated_sequence = sequence.copy()
    mutated_sequence[position] = nucleotide_map[new_base]
    return mutated_sequence

def evolve_sequence(sequence, model, iterations):
    """Evolve the sequence over a specified number of iterations."""
    top_scores = []
    top_sequences = []  # Keep as a list to avoid ragged arrays

    # Evaluate the starting sequence
    initial_prediction = model.predict(np.expand_dims(sequence, axis=0))[0, 0]
    top_scores.append(np.array([initial_prediction]))  # Convert to NumPy array
    top_sequences.append(sequence.copy())

    current_sequence = sequence.copy()

    for iteration in range(iterations):
        mutations = []

        # Generate all mutations for the current sequence
        for pos in range(current_sequence.shape[0]):
            for new_base in range(4):
                if not np.array_equal(current_sequence[pos], nucleotide_map[new_base]):
                    mutated_sequence = mutate_sequence(current_sequence, pos, new_base)
                    mutations.append(mutated_sequence)

        # Convert list of mutations to a consistent NumPy array
        mutations_array = np.stack(mutations)

        # Predict all mutations
        predictions = model.predict(mutations_array, batch_size=64)  # Adjust batch size as needed

        # Find the best mutation
        max_index = np.argmax(predictions)
        best_prediction = predictions[max_index]
        best_mutated_sequence = mutations_array[max_index]

        # Update the current sequence to the best mutation
        current_sequence = best_mutated_sequence.copy()

        # Store the best score and sequence
        top_scores.append(best_prediction)
        top_sequences.append(current_sequence.copy())

    # Convert top_scores to a flat NumPy array
    top_scores = np.array([float(score) for score in top_scores])

    return top_scores, top_sequences

# Initialize lists to store results
all_scores = []
all_sequences = []

# Iterate over each sequence in the list for 10 times
for sequence in seq_to_eval:
    # Evolve the current sequence
    scores, sequences = evolve_sequence(sequence, main_model, 10)
    
    # Store the results
    all_scores.append(scores)
    all_sequences.append(sequences)
