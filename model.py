import os
import torch
import torch.nn as nn
import joblib
import config

class EncoderModel_LSTM(nn.Module):
    def __init__(self):
        super(EncoderModel_LSTM, self).__init__()
        self.encoder = nn.LSTM(input_size=config.num_encoder_tokens, hidden_size=config.latent_dim, batch_first=True)
        
    def forward(self, encoder_inputs):
        encoder_outputs, (state_h, state_c) = self.encoder(encoder_inputs)
        return (state_h, state_c)


class DecoderModel_LSTM(nn.Module):
    def __init__(self):
        super(DecoderModel_LSTM, self).__init__()
        self.decoder = nn.LSTM(input_size=config.num_decoder_tokens, hidden_size=config.latent_dim, batch_first=True)
        self.decoder_dense = nn.Linear(config.latent_dim, config.num_decoder_tokens)
        
    def forward(self, decoder_inputs, encoder_states):
        decoder_outputs, state = self.decoder(decoder_inputs, encoder_states)
        final_outputs = self.decoder_dense(decoder_outputs)
        return final_outputs,state
    
class EncoderModel_GRU(nn.Module):
    def __init__(self):
        super(EncoderModel_GRU, self).__init__()
    
        self.encoder = nn.GRU(input_size=config.num_encoder_tokens, hidden_size=config.latent_dim, batch_first=True)
        
    def forward(self, encoder_inputs):

        encoder_outputs, state_h = self.encoder(encoder_inputs)
        return state_h  


class DecoderModel_GRU(nn.Module):
    def __init__(self):
        super(DecoderModel_GRU, self).__init__()
        self.decoder = nn.GRU(input_size=config.num_decoder_tokens, hidden_size=config.latent_dim, batch_first=True)
        self.decoder_dense = nn.Linear(config.latent_dim, config.num_decoder_tokens)
        
    def forward(self, decoder_inputs, encoder_states):

        decoder_outputs, state = self.decoder(decoder_inputs, encoder_states)
        final_outputs = self.decoder_dense(decoder_outputs)
        return final_outputs,state

import joblib
import os 
import numpy as np


    
def inference_model():
    # token_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/tokenizer_lstm_1500"
    # encoder_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/encoder_model_lstm.pth"
    # decoder_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/decoder_model_lstm.pth"

    # token_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/tokenizer_gru_1500"
    # encoder_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/encoder_model_gru.pth"
    # decoder_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/decoder_model_gru.pth"
    
    token_path_lstm = r"model_final_2/tokenizer_lstm_6000"
    encoder_path_lstm = r"model_final_2/encoder_model_lstm.pth"
    decoder_path_lstm = r"model_final_2/decoder_model_lstm.pth"

    token_path_gru = r"model_final_2/tokenizer_gru_6000"
    encoder_path_gru = r"model_final_2/encoder_model_gru.pth"
    decoder_path_gru = r"model_final_2/decoder_model_gru.pth"
    
    # token_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final_3/tokenizer_lstm_6000"
    # encoder_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final_3/encoder_model_lstm.pth"
    # decoder_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final_3/decoder_model_lstm.pth"

    # token_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final_3/tokenizer_gru_6000"
    # encoder_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final_3/encoder_model_gru.pth"
    # decoder_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final_3/decoder_model_gru.pth"
        
    with open(token_path_gru, 'rb') as file:
        tokenizer_gru = joblib.load(file)
    with open(token_path_lstm, 'rb') as file:
        tokenizer_lstm = joblib.load(file)
    encoder_lstm = EncoderModel_LSTM()
    decoder_lstm = DecoderModel_LSTM()
    encoder_gru = EncoderModel_GRU()
    decoder_gru = DecoderModel_GRU()
    encoder_lstm.load_state_dict(torch.load(encoder_path_lstm, map_location=torch.device('cpu')))
    encoder_lstm.eval() 
    decoder_lstm.load_state_dict(torch.load(decoder_path_lstm, map_location=torch.device('cpu')))
    decoder_lstm.eval()
    encoder_gru.load_state_dict(torch.load(encoder_path_gru, map_location=torch.device('cpu')))
    encoder_gru.eval() 
    decoder_gru.load_state_dict(torch.load(decoder_path_gru, map_location=torch.device('cpu')))
    decoder_gru.eval()
    return tokenizer_lstm, encoder_gru, decoder_gru,encoder_lstm,decoder_lstm
        
        
    # token_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/tokenizer_lstm_1500"
    # encoder_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/encoder_model_lstm.pth"
    # decoder_path_lstm = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/decoder_model_lstm.pth"

    # token_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/tokenizer_gru_1500"
    # encoder_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/encoder_model_gru.pth"
    # decoder_path_gru = r"C:/Users/HP/Desktop/one/T/Video-Captioning/model_final/decoder_model_gru.pth"