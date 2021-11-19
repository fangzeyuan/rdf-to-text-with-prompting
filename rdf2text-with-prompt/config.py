
class Config:
    # 0. gpu
    use_gpu = True
    gpu_id = "0"
    #device = torch.device("cuda:{}".format(gpu_id) if (torch.cuda.is_available() and use_gpu) else "cpu")

    flag = "finetune_gpt2"  ## finetune_gpt2, finetune_gpt2_additional_layer, finetune_gpt2_prefix, finetune_gpt2_vae
    ## Initialization parameters
    train_data_file = "data/e2e_data/src1_train.txt"
    eval_data_file = "data/e2e_data/src1_valid.txt"
    test_data_path = "data/e2e_data/src1_test.txt"
    model_type = 'gpt2'  # same as model_name_or_path
    checkpoint_num = "checkpoint-1132"
    dataset_path = 'data/e2e_data/'
    if flag == "finetune_gpt2":
        directory_name = "model_gpt2"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_additional_layer":
        directory_name = "model_gpt2_fc_layer_without_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_prefix":
        directory_name = "model_gpt2_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_vae":
        directory_name = "model_gpt2_vae_without_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_additional_layer_with_prefix":
        directory_name = "model_gpt2_fc_layer_with_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    elif flag == "finetune_gpt2_vae_with_prefix":
        directory_name = "model_gpt2_vae_with_prefix"
        output_dir = dataset_path+directory_name
        model_path = dataset_path+directory_name+"/"+checkpoint_num
        output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"test_result.csv"
        evaluate_output_path = dataset_path+directory_name+"/"+checkpoint_num+"/"+"evaluate_test_result.csv"
    num_train_epochs = '3'
    block_size = '512'
    save_steps = '2000'
    logging_steps = '1000'
    learning_rate = '0.001'
    gradient_accumulation_steps = '4'
    per_gpu_train_batch_size = '16'
    per_gpu_eval_batch_size = '16'
    sts_model = 'paraphrase-distilroberta-base-v1'

    ## The parameter for Generate function
    test_batch_size = 16
    min_length = 100
    max_length = 512
    do_sample = True
    top_p = 0.9
    top_k = 40
    num_return_sequences = 5
    # This configuration is used to define the number of predictions we want to evaluate on while probing bleu score
    # See generate.generate_predictions_for_bleu_score_probing(..) for details
    test_num_return_sequences = 2
    logging_steps_generate = 50

    ## The parameter for evaluate function
    logging_steps_evaluate = 50





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