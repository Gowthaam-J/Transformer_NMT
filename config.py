from pathlib import Path
import os
def get_config():
  return{
      "batch_size":32,
      "num_epochs":10,
      "lr":10**-4,
      "seq_len":128,
      "d_model":512,
      "lang_src":"en",
      "lang_tgt":"fr",
      "model_folder":"weights",
      "model_filename":"tmodel_",
      "preload":None,
      "tokenizer_file":"tokenizer_{0}.json",
      "experiment_name":"runs/tmodel"
  }

def get_weights_file_path(config,epoch:str):
  model_folder=f"{config['model_folder']}"
  model_filename=f"{config['model_filename']}{epoch}.pth"
  return os.path.join(model_folder,model_filename)