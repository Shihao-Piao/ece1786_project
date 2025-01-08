import torch
import torchtext
import gradio as gr
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
import torch
from models.transformer import SeqtoSeqTransformer
from utils.vocab_generator import VocabGenerator
import pandas as pd
from utils.utils import clean_eng_text,clean_ger_text
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

df_train = pd.read_csv('data/train_30.csv').loc[:60000]
df_train["English"] = df_train["English"].apply(lambda x: clean_eng_text(x))
df_train["Ger"] = df_train["Ger"].apply(lambda x: clean_ger_text(x))

#creating corpus
corpus_eng = ' '.join(list(df_train['English'].values))
vocab_eng = VocabGenerator(corpus=corpus_eng, min_frequency=2,tokenizer_lang="english")

corpus_ger = ' '.join(list(df_train['Ger'].values))
vocab_ger = VocabGenerator(corpus=corpus_ger, min_frequency=2,tokenizer_lang="germen")
# params
src_vocab_size = vocab_ger.__len__() - 58
target_vocab_size = vocab_eng.__len__() -2
max_len_src = 42
max_len_target = 42

embedding_size = 512
no_of_heads = 8
no_of_encoders = 3
no_of_decoders = 3
drop_out = 0.2
no_fwd_expansion = 1024
pad_idx = vocab_ger.stoi["<PAD>"]
device = 'cpu'
model = SeqtoSeqTransformer(src_vocab_size,target_vocab_size,max_len_src,max_len_target,embedding_size,
                 no_of_heads,no_of_encoders,no_of_decoders,pad_idx,
                 drop_out,no_fwd_expansion,device)
model_path= 'results/transformers/model_transformers.pt'
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
model.eval()



def predict_text(text, vocab_ger, vocab_eng, input_max_seq_length=23, target_max_seq_length=23):
    """
    function to predict the input text. It will translates the input german text to english
    Args:
        vocab_ger: german vocabulary
        vocab_eng: english vocabulary
        input_max_seq_length: maximum length of input sequence (Excluding EOS and SOS toekens)
        target_max_seq_length: maximum length of target sequence (Excluding EOS and SOS toekens)
    Returns:
        predicted sentance
    """
    text = clean_ger_text(str(text))
    # generate integers
    tokens = vocab_ger.generate_numeric_tokens(text)
    # add <sos> and <eos>
    tokens = vocab_ger.add_eos_sos(tokens)
    # padd
    tokens = vocab_ger.pad_sequence(tokens, max_seq_length=input_max_seq_length)
    ger_tensor = torch.tensor(tokens).unsqueeze(0)
    # move input tensor to device
    ger_tensor = ger_tensor.to(device)

    with torch.no_grad():
        encoder_out = model.encoder(ger_tensor)

    # label tensor. we will begin with <SOS>
    outputs = [vocab_eng.stoi["<SOS>"]]
    predicted_sentence = []

    for i in range(target_max_seq_length + 2):  # 1 for each sos and eos

        output_tok = torch.LongTensor(outputs).unsqueeze(0).to(device)
        # output_tok = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():

            out = model.decode(output_tok, encoder_out)
            out = model.fc(out)

        # finding the token integer
        out = out[:, -1, :]
        out = F.softmax(out)

        predicted_word_int = out.argmax(1).item()
        outputs.append(predicted_word_int)

        # if end of senetence break
        if predicted_word_int == vocab_eng.stoi["<EOS>"]:
            break

    # print(outputs)
    predicted_sentence = [vocab_eng.itos[word_int] for word_int in outputs]
    predicted_sentence = predicted_sentence[1:]
    predicted_sentence = ' '.join(predicted_sentence)

    pred = predicted_sentence.split()
    pred = ' '.join(pred[:-1])
    return pred

def generate_prediction(input_text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=40, num_beams=5, early_stopping=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def predict(sentence,choice=['both']):
    '''
    model_name = "neuesql/translate-de-en"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Define the input text in German
    input_text = sentence
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    # Generate the translation
    translation = model.generate(input_ids)
    # Decode the generated output into text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    # Print the translated text
    #print("Translated Text:", translated_text)
    '''
    translated_text = predict_text(sentence, vocab_ger, vocab_eng)
    # formal -> informal
    # Test the model
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.load_state_dict(torch.load('results/tune_model.pt'))
    model.to(device)
    informal_text = generate_prediction(translated_text, model, tokenizer, device)
    #print(informal_text)
    if len(choice) == 0:
        result = translated_text + '\n' + informal_text
    elif choice[0] == 'formal':
        result = "Formal:"+'\n'+translated_text
    elif choice[0] == 'informal':
        result = "Informal:"+'\n'+informal_text
    else:
        result = "Formal:"+'\n'+translated_text+'\n'+"Informal:"+'\n'+informal_text

    return result


if __name__ == '__main__':
    checkbox = gr.inputs.CheckboxGroup(
        label="Select one option",
        choices=["formal", "informal", "both"]
    )
    demo = gr.Interface(
        fn=predict,
        inputs=[gr.Textbox(placeholder="Write a sentence"),checkbox],
        outputs=["text"],
    )
    demo.launch(share=True)


"Ein alter Mann  der allein ein Bier trinkt."
