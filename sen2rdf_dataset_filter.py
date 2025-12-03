import pandas as pd
from config import opt


pd_reader = pd.read_csv(opt.evalaute_webnlg_output_files_sts_roberta_test_file)
full_rela_lst = []
full_src_lst = []
full_tgt_lst = []
edited_sents = []
special_tokens_dict = {'additional_special_tokens': '<rdf2sen>', 'bos_tok': '<|endoftext|>','eos_tok':'<|endoftext|>'}

for index, row in pd_reader.iterrows():
    str1 = str(row['bleu_pred'])
    str2 = str(row['rouge_pred'])
    if str1[0] != '|':
        str1 = ' | ' + str1
    if str2[0] != '|':
        str2 = ' | ' + str2
    if str1[0] == '|':
        str1 = ' ' + str1
    if str2[0] == '|':
        str2 = ' ' + str2

    str3 = row['input'].replace("<|endoftext|>","").strip()

    if str1 == str2 and str1 != 'nan' and str1 != 'None' and str1 != '':
        proc_sent = str1.split(" : ")
        if "prefix" in opt.flag:
            if len(proc_sent) == 3:
                full_rela_lst.append([proc_sent[1]])
                sent = ' {} {} {} '.format(special_tokens_dict["additional_special_tokens"], str1, special_tokens_dict["bos_tok"]) + str3 + ' {}'.format(special_tokens_dict["eos_tok"])
                edited_sents.append(sent)
                full_src_lst.append(str1)
                full_tgt_lst.append(str3)
        else:
            if len(proc_sent) == 3:
                full_rela_lst.append([proc_sent[1]])
                sent = ' {} {} '.format(str1, special_tokens_dict["bos_tok"]) + str3 + ' {}'.format(special_tokens_dict["eos_tok"])
                edited_sents.append(sent)
                full_src_lst.append(str1)
                full_tgt_lst.append(str3)

    elif str1 != str2 and str1 != 'nan' and str1 != 'None' and str1 != '':
        item_list = [str1, str2]
        for item in item_list:
            proc_sent = item.split(" : ")
            if "prefix" in opt.flag:
                if len(proc_sent) == 3:
                    full_rela_lst.append([proc_sent[1]])
                    sent = ' {} {} {} '.format(special_tokens_dict["additional_special_tokens"], item, special_tokens_dict["bos_tok"]) + str3 + ' {}'.format(special_tokens_dict["eos_tok"])
                    edited_sents.append(sent)
                    full_src_lst.append(item)
                    full_tgt_lst.append(str3)
            else:
                if len(proc_sent) == 3:
                    full_rela_lst.append([proc_sent[1]])
                    sent = ' {} {} '.format(item, special_tokens_dict["bos_tok"]) + str3 + ' {}'.format(special_tokens_dict["eos_tok"])
                    edited_sents.append(sent)
                    full_src_lst.append(item)
                    full_tgt_lst.append(str3)

test = pd.DataFrame({'edited_sents':edited_sents,'full_src_lst':full_src_lst,"full_tgt_lst":full_tgt_lst})
test.to_csv('data_filter.csv',index = None,encoding = 'utf8')