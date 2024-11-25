def predict (inf_encoder_model, inf_decoder_model, input_seq, tokenizer, max_decoder_seq_length):
#     with torch.no_grad():
#         h, c = inf_encoder_model(input_seq)
#     start_token = torch.zeros(1, 1, config.num_decoder_tokens)
#     decoded_sequence = []
#     states = (h, c)

#     for _ in range(max_decoder_seq_length):
#         with torch.no_grad():
#             # Pass the start token and states into the decoder
#             output= inf_decoder_model(start_token, states)
        
#         # Get the predicted token from the output
#         predicted_token = torch.argmax(output[0, -1, :]).item()
#         decoded_sequence.append(predicted_token)

#         # Update start token for the next time step
#         start_token = torch.zeros(1, 1, config.num_decoder_tokens)
#         start_token[0, 0, predicted_token] = 1.0  # Set the predicted token as the next input

#         # Stop decoding if we encounter the stop token
#         if predicted_token == tokenizer.word_index['<end>']:
#             break

#     # Convert predicted sequence of indices back to words
#     decoded_sentence = tokenizer.sequences_to_texts([decoded_sequence])
#     return decoded_sentence


# if __name__ == "__main__":
#     tokenizer, inf_encoder_model, inf_decoder_model = inference_model()