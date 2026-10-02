from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
from tensorflow.keras.models import load_model

## load the model
main_model = load_model('cl3_1fc_model')

## load positive test fasta file, without augmentation reverse complement
path='/project/bat_analysis/data/enhancer/prox_dist/bat/meis2/deepLearning/bat_model'
input_fasta_data = IOHelper.get_fastas_from_file(path + '/X.test.pos.genrich.fasta', uppercase=True)
seq_to_eval = SequenceHelper.do_one_hot_encoding(input_fasta_data.sequence, 401,
                                                      SequenceHelper.parse_alpha_to_seq)

## predictions
predicted_scores = main_model.predict(seq_to_eval)

## define shap score calculating function
def get_shap_explanations(seqs_to_explain):
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # this is required due to conflict between versions (https://github.com/slundberg/shap/issues/1110)
    dinuc_shuff_explainer = shap.DeepExplainer((main_model.layers[0].input, main_model.layers[-2].output), shuffle_several_times)
    raw_shap_explanations = dinuc_shuff_explainer.shap_values(seqs_to_explain, check_additivity=False)
    return raw_shap_explanations

## estimate hypothetical contribution scores
raw_shap_scores = get_shap_explanations(seq_to_eval)

## turn hypothetical contribution scores to actual contribution scores
final_contr_scores=raw_shap_scores*seq_to_eval

## save the contribution score results
import os
import h5py # https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html

out = '/project/bat_analysis/data/enhancer/prox_dist/bat/meis2/deepLearning/bat_model/interpretation/cl3_1fc_contr_scores.h5'
f = h5py.File(out, 'w')

# save the actual contribution scores
g = f.create_group("contrib_scores")
g.create_dataset('contrib_scores', data=final_contr_scores)

# save the hypothetical contribution scores
g = f.create_group("hyp_contrib_scores")
g.create_dataset('contrib_scores', data=raw_shap_scores)
f.close()
