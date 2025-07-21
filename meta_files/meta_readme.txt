1. 
The validation/test sets of the challenge have gone through a more careful quality check. 
for validation/test splits, plans are D05_div_D95 < 1.07 and max_div_D95 < 1.12. However, the number of D05_div_D95 and max_div_D95 is not disclosured to participants. 
D05_div_D95 = D05 / D95 of the PTV high. 
max_div_D95 = max(D) / D95 of the PTV high. 

2. 
The difference of "phase" and "dev_split" in meta_data.csv
"phase" indicates the data used for different phases of the challenge. 
For example, when phase == train, those data can be for training because the grouth truth is provided. If phase == valid, those data are only provided with input and no grouth truth. 
The data of final phase "test" of the challenge are completely hidden. 

"dev_split" is used for train the baseline model. The data with "dev_split == train" are used for model gradients update, and "dev_split == valid" are used for model selection. 
data of "dev_split == test" (i.e., "phase == valid") are kept untouched for baseline development. 