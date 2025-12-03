# README #
 #### `pip install -r requirements.txt`
 
## Dataset
Google Drive/dataset

## The current version from 2021/09/19
### Finetune GPT2 with four methods ##
You can change your hyperparameters in the config.py.
When you do finetune, make sure the `finetune(paras=paras)` in the main function is uncommented. You need to comment the lines (generate and evaluation) below the `finetune(paras=paras)`. 

### Generate GPT2 with four methods ##
You can change your hyperparameters in the config.py. Especially you need to set the `checkpoint_num` to a checkpoint you needed. For example, `checkpoint_num = "checkpoint-4000"`. Same as the evaluation.
When you do generate, make sure the `generate.generate(model_type=opt.model_type,
              model_path=opt.model_path,
              test_data_path=opt.test_data_path,
              batch_size=opt.batch_size,
              output_path=opt.output_path)` in the main function is uncommented. You need to comment the lines (finetune and evaluation) like the above. 

### Evaluate GPT2 with four methods ##
You can change your hyperparameters in the config.py.
When you do evaluate, make sure the `evaluation.batch_generate(
         input_path=opt.output_path,
         output_path=opt.evaluate_output_path,
         logging_steps=opt.logging_steps_evaluate
     )` in the main function is uncommented. You need to comment the lines (finetune and generate) like the above. 

1. finetune/generate/evaluate with sample and label
    #### Step 1: Change the flag = "finetune_gpt2" in config.py
    #### Step 2: python finetune.py
2. finetune/generate/evaluate with sample and prefix prompt and label
    #### Step 1: Change the flag = "finetune_gpt2_prefix" in config.py
    #### Step 2: python finetune.py
3. finetune/generate/evaluate with sample and dynamic token (from sample via pre-trained sts model) and label
    #### Step 1: Change the flag = "finetune_gpt2_additional_layer" in config.py
    #### Step 2: python finetune.py
4. finetune/generate/evaluate with sample and dynamic token (from sample via vae model) and label
    #### Step 1: Change the flag = "finetune_gpt2_vae" in config.py
    #### Step 2: python finetune.py

## The old version before 2021/09/19
### Finetune GPT2 with four methods ###
1. finetune with sample and label 
    ####  `finetune_gpt2.py`
2. finetune with sample and prefix prompt and label
    ####  `finetune_gpt2_prefix.py`
3. finetune with sample and dynamic token (from sample via pre-trained sts model) and label
    ####  `finetune_gpt2_additional_layer.py`
4. finetune with sample and dynamic token (from sample via vae model) and label
    ####  `finetune_gpt2_vae.py`