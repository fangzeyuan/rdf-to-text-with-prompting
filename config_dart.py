
class Config:
    # 0. gpu
    use_gpu = True
    gpu_id = "2"
    #device = torch.device("cuda:{}".format(gpu_id) if (torch.cuda.is_available() and use_gpu) else "cpu")

    flag = "finetune_gpt2"  ## finetune_gpt2, finetune_gpt2_additional_layer, finetune_gpt2_prefix, finetune_gpt2_vae
    ## Initialization parameters
    train_data_file = "data/DART/dart-v1.1.1-full-train.json"
    eval_data_file = "data/DART/dart-v1.1.1-full-dev.json"
    test_data_path = "data/DART/dart-v1.1.1-full-test.json"
    model_type = 't5-small'  # same as model_name_or_path
    checkpoint_num = "checkpoint-313320"
    dataset_path = 'data/DART/'  ## if the task is sen2rdf, set the dataset_path to 'data/DART_sen2rdf/'
    rdf2sen_augment_data_tag = False
    sen2rdf_augment_data_tag = False
    webnlg_rdf2sen_augment_data_tag = False
    webnlg_sen2rdf_augment_data_tag = False
    lower_threshold = 50
    upper_threshold = 60
    augment_tag = "sts-roberta"  # sts-roberta, vae
    paraphrased_tag = False
    if 't5' in model_type:
        directory_name = 't5_'
    else:
        directory_name = ''
    # directory_name += 'sen2rdf_'
    if sen2rdf_augment_data_tag:
        directory_name += 'augment_'
    if flag == "finetune_gpt2" and webnlg_rdf2sen_augment_data_tag == True and rdf2sen_augment_data_tag != True and paraphrased_tag == False:
        directory_name += "model_gpt2_aug_webnlg_"+str(lower_threshold)+"_"+str(upper_threshold)+"_"+augment_tag
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"  ## dataset_path + directory_name + "/" + checkpoint_num + "/" + "evaluate_total_test_result.csv"
    elif flag == "finetune_gpt2" and rdf2sen_augment_data_tag != True and paraphrased_tag == False:
        directory_name += "model_gpt2"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2" and rdf2sen_augment_data_tag == True and paraphrased_tag == False:
        directory_name += "model_gpt2_aug_total_"+str(lower_threshold)+"_"+str(upper_threshold) + "_" + augment_tag
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2" and paraphrased_tag == True and rdf2sen_augment_data_tag == False:
        directory_name += "model_gpt2_NLPAUG_noising_paraphrased"
        output_dir = dataset_path + directory_name
        model_path = dataset_path + directory_name + "/" + checkpoint_num
        output_path = dataset_path + directory_name + "/" + checkpoint_num + "/" + "test_result.csv"
        evaluate_output_path = dataset_path + directory_name + "/" + checkpoint_num + "/" + "evaluate_test_result.csv"
    elif flag == "finetune_gpt2_additional_layer":
        directory_name += "model_gpt2_fc_layer_without_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"dev_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_total_train_result.csv"
    elif flag == "finetune_gpt2_prefix":
        directory_name += "model_gpt2_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_vae":
        directory_name += "model_gpt2_vae_without_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_additional_layer_with_prefix":
        directory_name += "model_gpt2_fc_layer_with_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_vae_with_prefix":
        directory_name += "model_gpt2_vae_with_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    num_train_epochs = '30'
    block_size = '512'
    save_steps = '2000'
    logging_steps = '10'
    learning_rate = '0.00001'
    gradient_accumulation_steps = '4'
    per_gpu_train_batch_size = '6'
    per_gpu_eval_batch_size = '6'
    sts_model = 'paraphrase-distilroberta-base-v1'

    ## The parameter for Generate function
    test_batch_size = 16
    min_length = 10
    max_length = 100
    do_sample = True
    top_p = 0.9
    top_k = 40
    num_return_sequences = 5
    # This configuration is used to define the number of predictions we want to evaluate on while probing bleu score
    # See generate.generate_predictions_for_bleu_score_probing(..) for details
    test_num_return_sequences = 5
    logging_steps_generate = 50

    ## The parameter for evaluate function
    logging_steps_evaluate = 50
    dart_files_sts_roberta_test_file = "./data/DART/model_gpt2_fc_layer_without_prefix/checkpoint-83552/evaluate_total_train_result.csv"
    dart_files_vae_test_file = "./data/webnlg_challenge_2017/model_gpt2_vae/evaluate_test_result_train_vae.csv"
    dart_output_files_sts_roberta_test_file = "/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART_sen2rdf/model_gpt2/checkpoint-104440/sts_roberta_finetune_train_result_filter_total_"+str(lower_threshold)+"_"+str(upper_threshold)+".csv"
    evalaute_dart_output_files_sts_roberta_test_file = 'data/DART/aug/Text2Data/Our_model/1_out_of_5_and_1_out_of_10/evalaute_sts_roberta_finetune_train_result_filter_total_70_80.csv'
    dart_output_files_vae_test_file = "/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART_sen2rdf/model_gpt2/checkpoint-104440/vae_finetune_train_result_filter_"+str(lower_threshold)+"_"+str(upper_threshold)+".csv"
    evalaute_dart_output_files_vae_test_file = "/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/DART_sen2rdf/model_gpt2/checkpoint-104440/evalaute_vae_finetune_train_result_filter_"+str(lower_threshold)+"_"+str(upper_threshold)+".csv"

    evalaute_webnlg_output_files_sts_roberta_test_file = "/data/qbao775/rdf2textWithPrompt_zeyuan/rdf2text-with-prompt/data/webnlg_challenge_2017_sen2rdf/model_gpt2/checkpoint-27045/evalaute_sts_roberta_finetune_train_result_filter_total_" + str(lower_threshold) + "_" + str(upper_threshold) + ".csv"

    if augment_tag == 'vae' and 'e2e' in train_data_file:
        rdf2sen_aug_data_filter = 'e2e_rdf2sen_data_filter_vae_total_' + str(lower_threshold) + '_' + str(upper_threshold) + '.csv'
    elif augment_tag == 'sts-roberta' and 'e2e' in train_data_file:
        rdf2sen_aug_data_filter = 'e2e_rdf2sen_data_filter_sts_total_' + str(lower_threshold) + '_' + str(upper_threshold) + '.csv'
    elif augment_tag == 'vae' and 'dart' in train_data_file:
        rdf2sen_aug_data_filter = 'dart_rdf2sen_data_filter_vae_total_' + str(lower_threshold) + '_' + str(upper_threshold) + '.csv'
    elif augment_tag == 'sts-roberta' and 'dart' in train_data_file:
        rdf2sen_aug_data_filter = 'dart_rdf2sen_data_filter_sts_total_' + str(lower_threshold) + '_' + str(upper_threshold) + '.csv'
    elif augment_tag == 'vae' and 'webnlg' in train_data_file:
        rdf2sen_aug_data_filter = 'webnlg_rdf2sen_data_filter_vae_total_' + str(lower_threshold) + '_' + str(upper_threshold) + '.csv'
    elif augment_tag == 'sts-roberta' and 'webnlg' in train_data_file:
        rdf2sen_aug_data_filter = 'webnlg_rdf2sen_data_filter_sts_total_' + str(lower_threshold) + '_' + str(upper_threshold) + '.csv'

def parse(self, kwargs):
    '''
    user can update the default hyperparamter
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            raise Exception('opt has No key: {}'.format(k))
        setattr(self, k, v)

    print('*************************************************')
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print("{} => {}".format(k, getattr(self, k)))

    print('*************************************************')


Config.parse = parse
opt = Config()