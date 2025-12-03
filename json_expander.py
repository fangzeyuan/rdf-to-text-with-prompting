from config import opt
import json
with open(opt.train_data_file) as f:
    lines_dict = json.load(f)
    json.dumps(lines_dict, sort_keys=True, indent=4, separators=(', ', ': '))
    if 'webnlg' in opt.train_data_file:
        with open('data/webnlg_challenge_2017/train_paraphrased_expand.json', 'w') as f:
            json.dump(lines_dict, f, sort_keys=True, indent=4, separators=(', ', ': '))
    elif 'dart' in opt.train_data_file:
        with open('data/DART/dart-paraphrased-full-train-expand.json', 'w') as f:
            json.dump(lines_dict, f, sort_keys=True, indent=4, separators=(', ', ': '))