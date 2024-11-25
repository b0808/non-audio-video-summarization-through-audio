
import torch
import config
import model
import extract_features
import numpy as np
import functools
import operator

from gtts import gTTS
from googletrans import Translator
map_location=torch.device('cpu') 
dif = {"Hindi":'hi', "Marathi":'mr', "Kannada":'kn',"English":'en'}
def text_to_speech(text, output_file,model_lang):
    tts = gTTS(text=text, lang=dif[model_lang])
    tts.save(output_file)
    print(f"Audio saved as {output_file}")

def translate_to_hindi(text,model_lang):
    translator = Translator()
    if model_lang !="en":
        translated_text = translator.translate(text, src='en', dest=dif[model_lang]).text
    return translated_text

class VideoDescriptionRealTime(object):

    def __init__(self, config,model_option,search_type):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability
        self.model_option = model_option
        self.search_type = search_type

        # models
        self.tokenizer_gru, self.encoder_gru, self.decoder_gru,self.encoder_lstm,self.decoder_lstm = model.inference_model()
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        # self.search_type = config.search_type
        self.num = 0

    def greedy_search(self, input_seq):   
        start_token = torch.zeros(1, 1, config.num_decoder_tokens)
        start_token[0, 0, self.tokenizer_gru.word_index['bos']] = 1.0
        
        decoded_sequence = []
       
        if(self.model_option=="LSTM + GRU"):
            with torch.no_grad():
                states_gru = self.encoder_gru(input_seq)
                states_lstm = self.encoder_lstm(input_seq)
            for _ in range(15):  
                with torch.no_grad():
                    output_lstm, state_h_lstm = self.decoder_lstm(start_token, states_lstm)
                    output_gru, state_h_gru = self.decoder_gru(start_token, states_gru)
                states_lstm = state_h_lstm
                states_gru = state_h_gru
                predicted_token = torch.argmax((output_lstm[0, -1, :]+output_gru[0, -1, :])/2).item()
                if predicted_token == 0:
                    continue
                if predicted_token == self.tokenizer_gru.word_index['eos']:
                    break
                decoded_sequence.append(predicted_token)
                start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                start_token[0, 0, predicted_token] = 1.0
                
        if(self.model_option=="LSTM"):
            with torch.no_grad():
                states_lstm = self.encoder_lstm(input_seq)
            for _ in range(15):  
                with torch.no_grad():
                    output_lstm, state_h_lstm = self.decoder_lstm(start_token, states_lstm)
                states_lstm = state_h_lstm
                predicted_token = torch.argmax((output_lstm[0, -1, :])).item()
                if predicted_token == 0:
                    continue
                if predicted_token == self.tokenizer_gru.word_index['eos']:
                    break
                decoded_sequence.append(predicted_token)
                start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                start_token[0, 0, predicted_token] = 1.0
                
        if(self.model_option=="GRU"):
            with torch.no_grad():
                states_gru = self.encoder_gru(input_seq)
            for _ in range(15): 
                with torch.no_grad():
                    output_gru, state_h_gru = self.decoder_gru(start_token, states_gru)
                states_gru = state_h_gru
                predicted_token = torch.argmax((output_gru[0, -1, :])).item()
                
                if predicted_token == 0:
                    continue
                if predicted_token == self.tokenizer_gru.word_index['eos']:
                    break
                decoded_sequence.append(predicted_token)
                start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                start_token[0, 0, predicted_token] = 1.0
                
        inv_map = {v: k for k, v in self.tokenizer_gru.word_index.items()}
        decoded_sentence = ' '.join([inv_map[token] for token in decoded_sequence if token in inv_map])
        return decoded_sentence





  
    def beam_search(self, input_seq, beam_width=3, max_length=15):
        # Initialize start token with BOS token
        if self.model_option == "LSTM":
            start_token = torch.zeros(1, 1, config.num_decoder_tokens)
            start_token[0, 0, self.tokenizer_gru.word_index['bos']] = 1.0
            inv_map = {v: k for k, v in self.tokenizer_gru.word_index.items()}

            # Get initial encoder states
            try:
                with torch.no_grad():
                    states_lstm = self.encoder_lstm(input_seq)
            
            except Exception as e:
                print("Error initializing encoder states:", e)
                return ""

            # Initialize beams with the start token and zero score
            beams = [(start_token, [], 0.0, states_lstm)]
            
            # Iterate through each decoding step
            for step in range(max_length):
                new_beams = []
                
                for start_token, sequence, score, states_lstm in beams:
                    try:
                        # Perform decoding only if states are valid
                        if states_lstm is None:
                            raise ValueError("Decoding states are None; check state initialization.")
                        
                        # print(f"Step {step}: Decoding with start_token: {start_token.shape} - sequence: {sequence}")

                        with torch.no_grad():
                            output_lstm, state_h_lstm = self.decoder_lstm(start_token, states_lstm)
                        if output_lstm is None:
                            raise ValueError("Decoder did not return valid output. Check decoder configurations.")
                        
                        # print(f"Step {step}: output_lstm shape: {output_lstm.shape}")

                        # Update the states for the next step
                        states_lstm = state_h_lstm

                        # Get the top `beam_width` tokens and their probabilities
                        topk_probs, topk_indices = torch.topk(output_lstm[0, -1, :], beam_width)

                        # Expand each beam with the top predictions
                        for i in range(beam_width):
                            predicted_token = topk_indices[i].item()
                            token_prob = topk_probs[i].item()

                            # Stop if EOS token is reached
                            if predicted_token == self.tokenizer_gru.word_index['eos']:
                                new_beams.append((None, sequence + [predicted_token], score + token_prob, None))
                                # print(new_beams)
                                continue

                            # Prepare the new start token for the next step
                            new_start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                            new_start_token[0, 0, predicted_token] = 1.0
                            new_beams.append((new_start_token, sequence + [predicted_token], score + token_prob, states_lstm))
                    
                    except Exception as e:
                        print(f"Error during decoding at step {step}: {e}")
                        continue

                # Sort and retain top beams based on score
                beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

                # If all beams have ended, break
                if all(beam[0] is None for beam in beams):
                    break

            # Select the best sequence from the beams
            best_sequence = max(beams, key=lambda x: x[2])[1]
            decoded_sentence = ' '.join([inv_map[token] for token in best_sequence if token in inv_map and token != self.tokenizer_gru.word_index['eos']])
            return decoded_sentence
        if self.model_option == 'GRU':
            start_token = torch.zeros(1, 1, config.num_decoder_tokens)
            start_token[0, 0, self.tokenizer_gru.word_index['bos']] = 1.0
            inv_map = {v: k for k, v in self.tokenizer_gru.word_index.items()}

            # Get initial encoder states
            try:
                with torch.no_grad():
                    states_lstm = self.encoder_gru(input_seq)

                # Confirm encoder outputs are not None
                if states_lstm is None:
                    raise ValueError("Encoder did not return valid states. Ensure encoder outputs are correctly initialized.")
            
            except Exception as e:
                print("Error initializing encoder states:", e)
                return ""

            # Initialize beams with the start token and zero score
            beams = [(start_token, [], 0.0, states_lstm)]
            
            # Iterate through each decoding step
            for step in range(max_length):
                new_beams = []
                
                for start_token, sequence, score, states_lstm in beams:
                    try:
                        # Perform decoding only if states are valid
                        if states_lstm is None:
                            raise ValueError("Decoding states are None; check state initialization.")
                        
                        print(f"Step {step}: Decoding with start_token: {start_token.shape} - sequence: {sequence}")

                        with torch.no_grad():
                            output_lstm, state_h_lstm = self.decoder_gru(start_token, states_lstm)
                        if output_lstm is None:
                            raise ValueError("Decoder did not return valid output. Check decoder configurations.")
                        
                        # print(f"Step {step}: output_lstm shape: {output_lstm.shape}")

                        # Update the states for the next step
                        states_lstm = state_h_lstm

                        # Get the top `beam_width` tokens and their probabilities
                        topk_probs, topk_indices = torch.topk(output_lstm[0, -1, :], beam_width)

                        # Expand each beam with the top predictions
                        for i in range(beam_width):
                            predicted_token = topk_indices[i].item()
                            token_prob = topk_probs[i].item()

                            # Stop if EOS token is reached
                            if predicted_token == self.tokenizer_gru.word_index['eos']:
                                new_beams.append((None, sequence + [predicted_token], score + token_prob, None))
                                # print(new_beams)
                                continue

                            # Prepare the new start token for the next step
                            new_start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                            new_start_token[0, 0, predicted_token] = 1.0
                            new_beams.append((new_start_token, sequence + [predicted_token], score + token_prob, states_lstm))
                    
                    except Exception as e:
                        print(f"Error during decoding at step {step}: {e}")
                        continue

                # Sort and retain top beams based on score
                beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

                # If all beams have ended, break
                if all(beam[0] is None for beam in beams):
                    break

            # Select the best sequence from the beams
            best_sequence = max(beams, key=lambda x: x[2])[1]
            decoded_sentence = ' '.join([inv_map[token] for token in best_sequence if token in inv_map and token != self.tokenizer_gru.word_index['eos']])
            return decoded_sentence

        if self.model_option == 'LSTM + GRU':
            start_token = torch.zeros(1, 1, config.num_decoder_tokens)
            start_token[0, 0, self.tokenizer_gru.word_index['bos']] = 1.0
            inv_map = {v: k for k, v in self.tokenizer_gru.word_index.items()}

            # Get initial encoder states
            try:
                with torch.no_grad():
                    states_lstm = self.encoder_lstm(input_seq)
                    states_gru = self.encoder_gru(input_seq)

                # Confirm encoder outputs are not None
                if states_lstm is None:
                    raise ValueError("Encoder did not return valid states. Ensure encoder outputs are correctly initialized.")
            
            except Exception as e:
                print("Error initializing encoder states:", e)
                return ""

            # Initialize beams with the start token and zero score
            beams = [(start_token, [], 0.0, states_lstm,states_gru)]
            
            # Iterate through each decoding step
            for step in range(max_length):
                new_beams = []
                
                for start_token, sequence, score, states_lstm,states_gru in beams:
                    try:
                        # Perform decoding only if states are valid
                        if states_lstm is None:
                            raise ValueError("Decoding states are None; check state initialization.")
                        
                        # print(f"Step {step}: Decoding with start_token: {start_token.shape} - sequence: {sequence}")

                        with torch.no_grad():
                            output_lstm, state_h_lstm = self.decoder_lstm(start_token, states_lstm)
                            output_gru, state_h_gru = self.decoder_gru(start_token, states_gru)
                        if output_lstm is None:
                            raise ValueError("Decoder did not return valid output. Check decoder configurations.")
                        
                        # print(f"Step {step}: output_lstm shape: {output_lstm.shape}")

                        # Update the states for the next step
                        states_lstm = state_h_lstm
                        states_gru = state_h_gru

                        # Get the top `beam_width` tokens and their probabilities
                        topk_probs, topk_indices = torch.topk((output_lstm[0, -1, :]+output_lstm[0, -1, :])/2, beam_width)

                        # Expand each beam with the top predictions
                        for i in range(beam_width):
                            predicted_token = topk_indices[i].item()
                            token_prob = topk_probs[i].item()

                            # Stop if EOS token is reached
                            if predicted_token == self.tokenizer_gru.word_index['eos']:
                                new_beams.append((None, sequence + [predicted_token], score + token_prob, None,None))
                                # print(new_beams)
                                continue

                            # Prepare the new start token for the next step
                            new_start_token = torch.zeros(1, 1, config.num_decoder_tokens)
                            new_start_token[0, 0, predicted_token] = 1.0
                            new_beams.append((new_start_token, sequence + [predicted_token], score + token_prob, states_lstm,states_gru))
                    
                    except Exception as e:
                        print(f"Error during decoding at step {step}: {e}")
                        continue

                # Sort and retain top beams based on score
                beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

                # If all beams have ended, break
                if all(beam[0] is None for beam in beams):
                    break

            # Select the best sequence from the beams
            best_sequence = max(beams, key=lambda x: x[2])[1]
            decoded_sentence = ' '.join([inv_map[token] for token in best_sequence if token in inv_map and token != self.tokenizer_gru.word_index['eos']])
            return decoded_sentence

    
    
    def index_to_word(self):
        # inverts word tokenizer
        index_to_word = {value: key for key, value in self.tokenizer.word_index.items()}
        return index_to_word

    def main_test(self,video_path):
        model = extract_features.model_cnn_load()
        X_test = extract_features.extract_features(video_path, model)
        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(torch.tensor(X_test.reshape((-1, 80, 4096))))
        if self.search_type == 'beam':
            sentence_predicted = self.beam_search(torch.tensor(X_test.reshape((-1, 80, 4096))))
        return sentence_predicted