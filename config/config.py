import time

learning_rate = 1e-3
num_epochs = 10
num_folds = 5
batch_size = 32
dataset_type = 'MRI_Guys_multimodal'
is_multimodal = True
add_cc_loss = False
add_pretrain = True
if add_pretrain:
    suff1 = ''
else:
    suff1 = 'NP'
if add_cc_loss:
    suff = ''
else:
    suff = 'NL'
timestamp = time.strftime("%Y%m%d-%H%M%S")

model_filename = '../model_weights/' + f'{dataset_type}_{suff1}{suff}_{timestamp}_BS{batch_size}_EP{num_epochs}.pt' 
