import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from config import *
from dataset import *
from model import *
from train import *
from inference import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = get_config()

# Load tokenizer
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab())).to(device)

# Load trained weights
model_filename = 'D:\Transformer\weights\\tmodel_10.pth'
state = torch.load(model_filename, map_location=device)
model.load_state_dict(state['model_state_dict'])
model.eval()

def causal_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask

def encode_src_sentence(sentence, tokenizer_src, device, seq_len):
    tokens = tokenizer_src.encode(sentence)
    token_ids = tokens.ids
    pad_id = tokenizer_src.token_to_id("[PAD]")
    encoder_input = torch.full((1, seq_len), pad_id, dtype=torch.long)
    encoder_input[0, :len(token_ids)] = torch.tensor(token_ids)
    return encoder_input.to(device)

def get_logits_and_next_token(model, encoder_output, encoder_mask, decoder_input, tokenizer_tgt, device):
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
    out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
    proj = model.project(out[:, -1])  # logits of last token
    probs = torch.softmax(proj, dim=-1)
    top1 = torch.argmax(probs, dim=-1)
    return probs.detach().cpu().numpy()[0], top1.item()

st.title("Transformer Visual Word-by-Word Translation")

input_sentence = st.text_input("Enter a sentence to translate (English):", "I am Elon Musk")

if st.button("Translate"):
    st.write("Starting step-by-step translation...")

    seq_len = config['seq_len']
    max_len = config['seq_len']

    encoder_input = encode_src_sentence(input_sentence, tokenizer_src, device, seq_len)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(2).to(device)
    encoder_output = model.encode(encoder_input, encoder_mask)

    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    decoder_input = torch.tensor([[sos_idx]], device=device)

    full_translation = []

    for i in range(max_len):
        probs, next_token = get_logits_and_next_token(model, encoder_output, encoder_mask, decoder_input, tokenizer_tgt, device)
        vocab = list(tokenizer_tgt.get_vocab().keys())

        fig, ax = plt.subplots(figsize=(10, 5))
        top_k = 20
        top_indices = np.argsort(probs)[-top_k:][::-1]
        ax.bar([vocab[i] for i in top_indices], probs[top_indices])
        plt.xticks(rotation=45)
        plt.title(f"Step {i+1} prediction probabilities")
        st.pyplot(fig)
        time.sleep(0.8)

        predicted_word = tokenizer_tgt.id_to_token(next_token)
        full_translation.append(predicted_word)
        st.success(f"Predicted word: {predicted_word}")

        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == eos_idx:
            st.write("End of sentence reached.")
            break

    st.write("**Final Translation:**", " ".join(full_translation))











