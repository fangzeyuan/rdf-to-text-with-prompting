import pandas as pd
from config import opt

if opt.augment_tag == 'vae' and 'webnlg' in opt.train_data_file:
    pd_reader = pd.read_csv(opt.webnlg_files_vae_test_file)
elif opt.augment_tag == 'sts-roberta' and 'webnlg' in opt.train_data_file:
    pd_reader = pd.read_csv(opt.webnlg_files_sts_roberta_test_file)
elif opt.augment_tag == 'vae' and 'e2e' in opt.train_data_file:
    pd_reader = pd.read_csv(opt.e2e_files_vae_test_file)
elif opt.augment_tag == 'sts-roberta' and 'e2e' in opt.train_data_file:
    pd_reader = pd.read_csv(opt.e2e_files_sts_roberta_test_file)
elif opt.augment_tag == 'vae' and 'dart' in opt.train_data_file:
    pd_reader = pd.read_csv(opt.dart_files_vae_test_file)
elif opt.augment_tag == 'sts-roberta' and 'dart' in opt.train_data_file:
    pd_reader = pd.read_csv(opt.dart_files_sts_roberta_test_file)

data = []
if 'dart' in opt.train_data_file or 'webnlg' in opt.train_data_file:
    for index, row in pd_reader.iterrows():
        if row['bleu_score'] >= opt.lower_threshold and row['bleu_score'] <= opt.upper_threshold and row['bleu_pred'].endswith('.'):
            data.append([row['input'],row['label'],row['bleu_pred'],row['bleu_score'],row['rouge_pred'],row['rouge_score']])

elif 'e2e' in opt.train_data_file:
    for index, row in pd_reader.iterrows():
        if row['bleu_score'] >= opt.lower_threshold and row['bleu_score'] <= opt.upper_threshold:
            data.append([row['input'],row['label'],row['bleu_pred'],row['bleu_score'],row['rouge_pred'],row['rouge_score']])

df = pd.DataFrame(data,columns=['input','label','bleu_pred','bleu_score','rouge_pred','rouge_score'])
df.to_csv(opt.rdf2sen_aug_data_filter,index = None,encoding = 'utf8')