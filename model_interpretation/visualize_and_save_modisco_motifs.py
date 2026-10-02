import h5py
import numpy as np
from collections import Counter
from modisco.visualization import viz_sequence
from matplotlib import pyplot as plt
import seaborn as sns

def write_meme_file(motifs, output_path, background=[0.27, 0.23, 0.23, 0.27]):
    with open(output_path, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies:\n")
        f.write("A {0:.3f} C {1:.3f} G {2:.3f} T {3:.3f}\n\n".format(*background))
        
        for motif_name, pwm in motifs.items():
            f.write("MOTIF {}\n".format(motif_name))
            f.write("letter-probability matrix: alength= 4 w= {}\n".format(len(pwm)))
            for row in pwm:
                f.write("{} {} {} {}\n".format(*row))
            f.write("\n")

hdf5_results = h5py.File("/project/bat_analysis/data/enhancer/prox_dist/bat/meis2/deepLearning/bat_model/interpretation/tfmodisco_results_cl3_1fc.pkl","r")

print("Metaclusters heatmap")
activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
                    np.array(
        [x[0] for x in sorted(
                enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
               key=lambda x: x[1])])]
sns.heatmap(activity_patterns, center=0)
plt.show()

metacluster_names = [
    x.decode("utf-8") for x in 
    list(hdf5_results["metaclustering_results"]
         ["all_metacluster_names"][:])]

all_motifs = {}

for metacluster_name in metacluster_names:
    print(metacluster_name)
    metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                   [metacluster_name])
    print("activity pattern:",metacluster_grp["activity_pattern"][...])
    all_pattern_names = [x.decode("utf-8") for x in 
                         list(metacluster_grp["seqlets_to_patterns_result"]
                                             ["patterns"]["all_pattern_names"][...])]
    if len(all_pattern_names) == 0:
        print("No motifs found for this activity pattern")
        continue
    
    for pattern_name in all_pattern_names:
        print(metacluster_name, pattern_name)
        pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
        print("total seqlets:", len(pattern["seqlets_and_alnmts"]["seqlets"]))
        
        forward_pwm = np.array(pattern["sequence"]["fwd"])
        all_motifs[f"{metacluster_name}_{pattern_name}"] = forward_pwm
        
        background = np.array([0.27, 0.23, 0.23, 0.27])
        print("Hypothetical scores:")
        viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"])
        print("Actual importance scores:")
        viz_sequence.plot_weights(pattern["task0_contrib_scores"]["fwd"])
        print("onehot, fwd and rev:")
        viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                        background=background)) 
        viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                        background=background))
        viz_sequence.plot_weights(pattern["sequence"]["fwd"]) 
        viz_sequence.plot_weights(pattern["sequence"]["rev"]) 
        
hdf5_results.close()

output_meme_path = "tfmodisco_motifs_meis_binding_model.meme"
write_meme_file(all_motifs, output_meme_path)
print(f"Motifs saved to {output_meme_path}")
