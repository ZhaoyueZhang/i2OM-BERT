# i2OM-BERT
A DNABERT-based tool for 2’-O-methylation (2OM) site identification.

Dataset:
  The Dataset for 2OM (2OM-adenine (A), cytosine (C), guanine (G), and uracil (U)) were retrieved from RMBase and experimental data by using Nm-seq (GEO Accession: GSE90164). There are 1205 positive samples and 1210 negative samples in 2OM-A dataset, 996 positive samples and 998 negative samples in 2OM-G dataset, 1172 positive samples and 1174 negative samples in 2OM-C dataset, 624 positive samples and 626 negative samples in 2OM-T dataset.
  The four subsets were randomly splits 4:1 for training and testing. In training, stratified 10 fold cross-validation was used for tuning the hyper-parameters.

Framework:
  
Result:
  In cross-validation, the AUROC for 2OM-Am, 2OM-Gm, 2OM-Gm and 2OM-Tm are , , , and ,respectively.
  On the testing set, the AUROC achieved , , , and  for 2OM-Am, 2OM-Gm, 2OM-Gm and 2OM-Tm,respectively.

Prediction Tool:

It is hoped that this tool can be useful for researches.
