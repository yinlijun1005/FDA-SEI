# FDA-SEI
Paper: L. Yin et al., "Few-Shot Domain Adaption-Based Specific Emitter Identification Under Varying Modulation," 2023 IEEE 23rd International Conference on Communication Technology (ICCT), Wuxi, China, 2023, pp. 1439-1443, doi: 10.1109/ICCT59356.2023.10419733.
Address:https://ieeexplore.ieee.org/document/10419733
# Requirement
Pytorch 1.10.1 with Python 3.6.15
# Code introduction
complexcnn.py,model_complexcnn_onlycnn.py  --> files for model
get_dataset_sourcetrain.py  --> file for source training data
get_dataset_transferlearning.py  --> file for transfer learning data
MMD_loss.py  --> MMD loss function
SourceTrain.py  --> file for source training
TL_MMD.py  --> file for transfer learning

You need to run SourceTrain.py to get the source domain model, then TL_MMD.py to get the transfer learning result.



