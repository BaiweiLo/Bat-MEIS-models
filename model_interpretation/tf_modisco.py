import h5py
import numpy as np
import modisco

## define tf modisco parameters
def run_tfmodisco1(onehot_seqs, contrib_scores):
    # Initialize the TF-MoDISco workflow
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        # Customizable parameters for TF-MoDISco
        sliding_window_size=15,  # Adjust based on your expected motif size
        flank_size=5,
        max_seqlets_per_metacluster=50000,
        target_seqlet_fdr=0.2,  # False discovery rate for seqlet identification
        seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            n_cores=4,  # Parallelization (adjust based on hardware)
            trim_to_window_size=15,
            initial_flank_to_add=5,
            kmer_len=6,  # K-mer length for motif discovery
            num_gaps=1,  # Number of gaps allowed in k-mers
            num_mismatches=1,  # Number of mismatches allowed
        )
    )

    # Run the TF-MoDISco workflow
    results = tfmodisco_results(
        task_names=["task0"],  # Assuming one task, adjust for multiple
        contrib_scores={'task0': contrib_scores},  # Contribution scores
        hypothetical_contribs={'task0': contrib_scores},  # USe actual contribution score for both
        one_hot=onehot_seqs  # One-hot encoded sequences
    )
    
    return results

# read contribution scores
out = '/project/bat_analysis/data/enhancer/prox_dist/bat/meis2/deepLearning/bat_model/interpretation/cl3_1fc_contr_scores.h5'
f = h5py.File(out,"r")

actual_score = np.array(f["contrib_scores"]['contrib_scores'])
modisco_results = run_tfmodisco1(seq_to_eval, actual_score[0])

## save tf modisco results
import modisco.util
! [[ -e results.hdf5 ]] && rm results.hdf5
grp = h5py.File("/project/bat_analysis/data/enhancer/prox_dist/bat/meis2/deepLearning/bat_model/interpretation/tfmodisco_results_cl3_1fc.pkl", "w")
modisco_results.save_hdf5(grp)

