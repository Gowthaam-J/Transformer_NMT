from config import *
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import *
from model import *
from train import *
# Load config
config = get_config()

tokenizer_src = Tokenizer.from_file('D:\Transformer\\tokenizer_en.json')
tokenizer_tgt = Tokenizer.from_file('D:\Transformer\\tokenizer_fr.json')

# Build the model
model = get_model(
    config,
    src_vocab_size=len(tokenizer_src.get_vocab()),
    tgt_vocab_size=len(tokenizer_tgt.get_vocab())
)

# Define the device before loading the checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('D:\Transformer\weights\\tmodel_10.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Load saved weights
model.eval()
model.to(device) 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
conig=get_config()
train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)
model=get_model(config,len(tokenizer_src.get_vocab()),len(tokenizer_tgt.get_vocab())).to(device)
model_filename=get_weights_file_path(config,1)
state=torch.load('D:\Transformer\weights\\tmodel_10.pth') #model_filename
model.load_state_dict(state['model_state_dict'])