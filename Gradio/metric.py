import openai
import torch
import torchtext
import gradio as gr
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from models.transformer import SeqtoSeqTransformer
from utils.vocab_generator import VocabGenerator
import pandas as pd
from utils.utils import clean_eng_text,clean_ger_text
import torch.nn.functional as F
import warnings
warnings.simplefilter("ignore")

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

def predict(sentence):
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
    tokenizer = AutoTokenizer.from_pretrained("s-nlp/t5-informal")
    model = AutoModelForSeq2SeqLM.from_pretrained("s-nlp/t5-informal")
    model.load_state_dict(torch.load('results/tune_model.pt'))
    model.to(device)
    informal_text = generate_prediction(translated_text, model, tokenizer, device)
    if translated_text == informal_text[:-1] :
        model_name = "s-nlp/t5-informal"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        input_text = translated_text
        # Tokenize the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate informal text
        output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)

        # Decode and print the generated text
        informal_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return informal_text,translated_text
df_test = pd.read_csv('data/test.csv')
gemany = df_test['Ger'].loc[:50].values
'''
gemany = [
'Ein Mann mit einem orangefarbenen Hut  der etwas anstarrt.',
'Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun.',
'Ein Mädchen in einem Karateanzug bricht ein Brett mit einem Tritt.',
'Fünf Leute in Winterjacken und mit Helmen stehen im Schnee mit Schneemobilen im Hintergrund.',
'Leute Reparieren das Dach eines Hauses.',
'Ein hell gekleideter Mann fotografiert eine Gruppe von Männern in dunklen Anzügen und mit Hüten  die um eine Frau in einem trägerlosen Kleid herum stehen.',
'Eine Gruppe von Menschen steht vor einem Iglu.',
'Ein Junge in einem roten Trikot versucht  die Home Base zu erreichen  während der Catcher im blauen Trikot versucht  ihn zu fangen.',
'Ein Typ arbeitet an einem Gebäude.',
'Ein Mann in einer Weste sitzt auf einem Stuhl und hält Magazine.'
]
'''
import time
count = 0
for i in range(len(gemany)):
  time.sleep(2)
  input_text,eng_text = predict(gemany[i])
  response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {
      "role": "system",
      "content": "Judge whether the input sentence is formal or informal. If the sentence is formal, print out 0; if the sentence is informal, print out 1."
    },
    {
      "role": "user",
      "content": input_text
    }
  ],
  temperature=1,
  max_tokens=1000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
  print("Output:",input_text)
  if int(response["choices"][0]["message"]["content"]) == 1:
      print('Informal')
      count += 1
  else:
    print('Formal')
  print('\n')

print(count/len(gemany))
