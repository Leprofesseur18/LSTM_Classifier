# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 22:19:49 2026

@author: dougl
"""

# Comando Linux simulam um ambiente para virtualmente carregarmos o modelo ollama
# o comando %%capture faz com que a execuçãio da célula não seja exibida

%%capture

! sudo apt update && sudo apt install pciutils lshw

!curl -fsSL https://ollama.com/install.sh | sh

!nohup ollama serve > ollama.log 2>&1 &
# %%
# Instalando o ollama

%%capture

!pip install ollama

import ollama
# %%
# Carregamento de todas bibliotecas para uso do modelo

%%capture

import pandas as pd
import numpy as np
import re
import ollama
from tqdm import tqdm
!pip install openpyxl
!pip install tensorflow

#Carregando os algoritmos

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#Funções para carregamento do modelo BERTimbau
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Curva ROC
from sklearn.metrics import roc_curve, auc,f1_score
import matplotlib.pyplot as plt
# %%
%%capture

# Carrega o modelo BERTimbau Base
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Função para gerar embeddings
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Pega o embedding da primeira posição [CLS]
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

# Função para calcular similaridade coseno
def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2).item()
# %%
# Rodando o modelo gemma, com 4 bilhões de parâmetros

ollama.pull("gemma3:4b-it-qat")

ollama.pull("gemma3:12b-it-qat")
# %%
# Função para limpar possíveis índices no início das frases
# e outros caracteres indesejados como aspas duplas, chaves, colchetes e parênteses

def remove_list_markers(text):
    # Use a regular expression to match and remove patterns like '1.' or '1. ' at the beginning of the string
    text = re.sub(r'^\d+\.\s*', '', text)
    # Remove double quotes, curly braces, square brackets, and parentheses
    text = re.sub(r'[\\"{}\[\]()]', '', text)
    return text
# %%
# Importanto o arquivo com as frases originais

file_path = '/content/drive/MyDrive/Bases/FakeNews_FrasesOriginais.xlsx'

try:
    frases_Fake = pd.read_excel(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    print("Please check the file path and make sure you have mounted your Google Drive correctly.")
# %%
    # criando a lista para receber dados limpos
lista1=[]
# %%
# Removendo sinais e aspas das frases, para facilitar o trabalho da rede gemma3 4b


for i in frases_Fake['frases']:
    frase=remove_list_markers(i)
    lista1.append(frase)
# %%

frases_Fake['frases']=lista1

print(frases_Fake)
# %%
#Criação de 15 paráfrases para cada frase original
armazenamento=[]
for i in tqdm(frases_Fake["frases"]):
  try:
    prompt_sys=("Você deve atuar como um especialista em linguística."
               "Como especialista em linguística, você terá de criar novas frases, por meio de uma que será apresentada,mantendo o sentido semântico")

    prompt_user= ( "Não escreva texto introdutório. "
                  "Não utilize números, índices, marcadores, ou símbolos como - ou *. "
                  "Não traga marcadores de lista"
                  "Escreva a paráfrase com um número de palavras próximo ao da frase original."
                  f"Escreva 15 paráfrases para: {i} ")
    response = ollama.chat(model='gemma3:4b-it-qat', messages=[

      {'role': 'system', 'content':prompt_sys},
      {'role': 'user', 'content':prompt_user}
       ],
        options={
            "temperature": 0.35,
            "top_k": 40,
            "top_p": 0.95
        }
    )
    content = response['message']['content']
    linhas = [line.strip() for line in content.split('\n') if line.strip()]

        # Store each paraphrase pair with the original phrase
    for paraphrase in linhas[:15]:  # Take first two paraphrases if more are generated
        armazenamento.append({
              "FraseOriginal": i,
              "Parafrase": paraphrase
            })
  except Exception as e:
        print(f"Error processing phrase: {i} - {str(e)}")
        # Add original phrase with error marker if needed
        armazenamento.append({
            "FraseOriginal": i,
            "Afirmacao": "[ERROR IN PROCESSING]"
        })
# %%
frasesFake=pd.DataFrame(armazenamento)

auxlista1=[]

print(frasesFake)
# %%
# Limepza das frases

for _,row in tqdm(frasesFake.iterrows()):
  FraseOriginal=row['FraseOriginal']
  Parafrase=row['Parafrase']

  FraseOriginal=remove_list_markers(FraseOriginal)
  Parafrase=remove_list_markers(Parafrase)

  auxlista1.append(FraseOriginal)
  auxlista1.append(Parafrase)
# %%
#Criação do Datafre

novasfrases=pd.DataFrame({"FraseOriginal":auxlista1[::2],"Parafrase":auxlista1[1::2]})

print(novasfrases)
# %%
# Criação da lita temporária similaridades

similaridades=[]
# %%
# Validação de similaridade com LLM - gemma3:12b-it-qat

# Validação de similaridade semantica das frasaes

for _, row in tqdm(novasfrases.iterrows()):
    i = row['FraseOriginal']
    ii = row['Parafrase']


    try:
        prompt_sys = (
            "Você é um especialista em linguística. "
            "Sua única tarefa é analisar duas frases e responder apenas com '1' se forem semanticamente similares, "
            "ou '0' se não forem."
        )

        prompt_user = (
            f"Frase 1: {i}\n"
            f"Frase 2: {ii}\n\n"
            "As frases têm o mesmo sentido semântico? "
            "Responda apenas com 1 (similares) ou 0 (diferentes)."
        )

        response = ollama.chat(
            model="gemma3:12b-it-qat",
            messages=[
                {'role': 'system', 'content': prompt_sys},
                {'role': 'user', 'content': prompt_user}
            ]
        )

        content = response['message']['content']

        # Extrai apenas 0 ou 1
        match = re.search(r'\b[01]\b', content)
        result = match.group(0) if match else "0"  # se não encontrar, assume 0

        similaridades.append({
            "FraseOriginal": i,
            "Parafrase": ii,
            "Similaridade": result
        })

    except Exception as e:
        print(f"Erro ao processar frase: {i} - {str(e)}")
        similaridades.append({
            "FraseOriginal": i,
            "Parafrase": ii,
            "Similaridade": "[ERROR]"
        })
# %%
# Criação de dataframe com silimaridade medida pelo gemma3:12b-it-qat

novasfrases2=pd.DataFrame(similaridades)

print(novasfrases2)
# %%
# Análise da Similaridade - Etapa 6

contagem=novasfrases2['FraseOriginal'].count()
Similares=novasfrases2[novasfrases2['Similaridade']=='1'].count()
Não_Similares=novasfrases2[novasfrases2['Similaridade']=='0'].count()
Porcentagem=Similares/contagem*100


valores=pd.DataFrame({"Total":contagem,"Similares":Similares,"Não Similares":Não_Similares,"Porcentagem":Porcentagem})

print(valores)
# %%
# Baixar o dataset - opicional

novasfrases.to_excel("novasfrases.xlsx", index=False)
from google.colab import files
files.download("novasfrases.xlsx")
# %% teste de etapa
carregamento=[]

tt=novasfrases.iloc[148:156]
# %%
for _,row in tqdm(novasfrases2.iterrows()):
  FraseOriginal=row['FraseOriginal']
  Parafrase=row['Parafrase']
  Similaridade=row['Similaridade']
  try:
    response = ollama.chat(model='gemma3:4b-it-qat', messages=[
      {'role': 'user', 'content':
       "Não utilize números, índices, marcadores ou símbolos como - ou *. "
       "Não escreva texto introdutório. "
       f"Reescreva a frase a seguir com sentido semântico oposto,sem exagero.Esta é a frase: {Parafrase}"

}],
         options={
            "temperature": 0.5,
            "top_k": 40,
            "top_p": 0.95
         }
    )
    #print(response['message']['content'])
    response['message']['content']
    liness = [line.strip() for line in response['message']['content'].split('\n') if line.strip()]

    # Store each paraphrase pair with the original phrase
    for ii in liness[:1]:  # Take first phrases if more are generated
        carregamento.append({
              "FraseOriginal": FraseOriginal,
              "Parafrase": Parafrase,
              "Similaridade_Parafrase_Original": Similaridade,
              "Parafrase_oposta":ii
            })

  except Exception as e:
        print(f"Error processing phrase: {i} - {str(e)}")
        # Add original phrase with error marker if needed
        carregamento.append({
            "FraseOriginal": i,
            "Afirmacao": "[ERROR IN PROCESSING]"
        })

# %%
# Organizando e criando o dataset com a versão antagõnica da frase, ou seja, a notíca verdadeira
novasfrasess=pd.DataFrame(carregamento)

print(novasfrasess)
# %%
# lista auxiliar carregamento2

carregamento2=[]
# %%

# Análise semântica das paráfrases opostas

# Análise do sentido semântico entre as paráfrases e paráfrases opostas

for _, row in tqdm(novasfrasess.iterrows()):
    FraseOriginal = row['FraseOriginal']
    Parafrase = row['Parafrase']
    Similaridade_Parafrase_Original = row['Similaridade_Parafrase_Original']
    Parafrase_oposta = row['Parafrase_oposta']

    try:
        prompt_sys = (
            "Você é um especialista em linguística. "
            "Sua única tarefa é analisar duas frases e responder apenas com '1' se forem semanticamente similares, "
            "ou '0' se não forem."
        )

        prompt_user = (
            f"Frase 1: {Parafrase}\n"
            f"Frase 2: {Parafrase_oposta}\n\n"
            "As frases têm o mesmo sentido semântico? "
            "Responda apenas com 1 (similares) ou 0 (diferentes)."
        )

        response = ollama.chat(
            model="gemma3:12b-it-qat",
            messages=[
                {'role': 'system', 'content': prompt_sys},
                {'role': 'user', 'content': prompt_user}
            ]
        )

        content = response['message']['content']

        # Extrai apenas 0 ou 1
        match = re.search(r'\b[01]\b', content)
        result = match.group(0) if match else "0"  # se não encontrar, assume 0

        carregamento2.append({
            "FraseOriginal": FraseOriginal,
            "Parafrase": Parafrase,
            "Similaridade_Parafrase_Original": Similaridade_Parafrase_Original,
            "Parafrase_oposta": Parafrase_oposta,
            "Similaridade_paraf_oposta": result
        })

    except Exception as e:
        print(f"Erro ao processar frase: {i} - {str(e)}")
        similaridades.append({
            "FraseOriginal": i,
            "Parafrase": ii,
            "Similaridade": "[ERROR]"
        })
# %%

# Criação do datafrme refrente a etapa anterior

novasfrasess=pd.DataFrame(carregamento2)

print(novasfrasess)
# %%

contagem=novasfrasess['FraseOriginal'].count()
Similares=novasfrasess[novasfrasess['Similaridade_paraf_oposta']=='1'].count()
Não_Similares=novasfrasess[novasfrasess['Similaridade_paraf_oposta']=='0'].count()
Porcentagem=Não_Similares/contagem*100


valores2=pd.DataFrame({"Total":contagem,"Similares":Similares,"Não Similares":Não_Similares,"Porcentagem":Porcentagem})

print(valores2)
# %%

carregamento3=[]
# %%

# criando a negação das frases originais


for ii in tqdm(frases_Fake["frases"]):
    response = ollama.chat(model='gemma3:4b-it-qat', messages=[
      {'role': 'user', 'content':
      "Não utilize números, índices, marcadores ou símbolos como - ou *. "
      "Não escreva texto introdutório. "
      f"Reescreva a frase a seguir com sentido semântico oposto,sem exagero.Esta é a frase: {ii}"

}],
    options={
            "temperature": 0.5,
            "top_k": 40,
            "top_p": 0.95
    }

     )
    #print(response['message']['content'])
    response['message']['content']
    liness = [line.strip() for line in response['message']['content'].split('\n') if line.strip()]
    carregamento3.append({"FraseOriginal":ii,"Parafrase_oposta":liness})
# %%

# Dataframe da erapa anterior

frases_Fake_comOpostas=pd.DataFrame(carregamento3)

print(frases_Fake_comOpostas)
# %%

# criando lista auxiliar para próxima etapa

aux=[]

print(aux)
# %%

# Removendo sinais e aspas das frases, para facilitar o trabalho da rede gemma3 4b

for _, row in tqdm(frases_Fake_comOpostas.iterrows()):
    i=remove_list_markers(row['FraseOriginal'])
    # Extract the single string from the list in 'Parafrase_oposta'
    cleaned_paraphrase = remove_list_markers(row['Parafrase_oposta'][0])
    aux.append({"FraseOriginal":i,"Parafrase_oposta":cleaned_paraphrase})
# %%

print(aux)
# %%

# Criação dataframe da etapa anterior

frases_Fake_comOpostas=pd.DataFrame(aux)

print(frases_Fake_comOpostas)
# %%

aux2=[]
# %%

# Similaridade entre as frases originais e as frases de oposição geradas de forma sintética

for _, row in tqdm(frases_Fake_comOpostas.iterrows()):
    FraseOriginal = row['FraseOriginal']
    Parafrase_oposta = row['Parafrase_oposta']

    try:
        prompt_sys = (
            "Você é um especialista em linguística. "
            "Sua única tarefa é analisar duas frases e responder apenas com '1' se forem semanticamente similares, "
            "ou '0' se não forem."
        )

        prompt_user = (
            f"Frase 1: {FraseOriginal}\n"
            f"Frase 2: {Parafrase_oposta}\n\n"
            "As frases têm o mesmo sentido semântico? "
            "Responda apenas com 1 (similares) ou 0 (diferentes)."
        )

        response = ollama.chat(
            model="gemma3:12b-it-qat",
            messages=[
                {'role': 'system', 'content': prompt_sys},
                {'role': 'user', 'content': prompt_user}
            ]
        )

        content = response['message']['content']

        # Extrai apenas 0 ou 1
        match = re.search(r'\b[01]\b', content)
        resultado = match.group(0) if match else "0"  # se não encontrar, assume 0

        aux2.append({
            "FraseOriginal": FraseOriginal,
            "Parafrase_oposta": Parafrase_oposta,
            "Similaridade_Parafrase_Original": resultado,

        })

    except Exception as e:
        print(f"Erro ao processar frase: {i} - {str(e)}")
        similaridades.append({
            "FraseOriginal": i,
            "Parafrase": ii,
            "Similaridade": "[ERROR]"
        })
# %%

#data frame etapa anterior

frases_Fake_comOpostas=pd.DataFrame(aux2)

print(frases_Fake_comOpostas)
# %%

contagem=frases_Fake_comOpostas['FraseOriginal'].count()
Similares=frases_Fake_comOpostas[frases_Fake_comOpostas['Similaridade_Parafrase_Original']=='1'].count()
Não_Similares=frases_Fake_comOpostas[frases_Fake_comOpostas['Similaridade_Parafrase_Original']=='0'].count()
Porcentagem=Similares/contagem*100


valores3=pd.DataFrame({"Total":contagem,"Similares":Similares,"Não Similares":Não_Similares,"Porcentagem":Porcentagem})

print(valores3)
# %%

# Criação de dataset Final

# Concatenação de todos os datasets

part1=pd.DataFrame({"frases":frases_Fake_comOpostas["FraseOriginal"],"Fake":1})
part2=pd.DataFrame({"frases":frases_Fake_comOpostas["Parafrase_oposta"],"Fake":0})

part3=pd.DataFrame({"frases":novasfrasess["Parafrase"],"Fake":1})
part4=pd.DataFrame({"frases":novasfrasess["Parafrase_oposta"],"Fake":0})

dfFinal=pd.concat([part1,part2,part3,part4],axis=0)
print(dfFinal)

# %%

#Balanceamento

observacoes=dfFinal['frases'].count()

observacoesFake=dfFinal[dfFinal['Fake']==1].count()
observacoesNaoFake=dfFinal[dfFinal['Fake']==0].count()
Porcentagem=observacoesFake/observacoes*100

valores

valores5=pd.DataFrame({"Total":observacoes,"Similares":observacoesFake,"Não Similares":observacoesNaoFake,"Porcentagem":Porcentagem})

print(valores5)

# %%

# Tokenização e padding das observações

tokenizer = Tokenizer(num_words=5000) # num_words mantem as palavras mais frequentes. Neste caso, as 5000 mais frequentes
tokenizer.fit_on_texts(dfFinal['frases'])
sequences = tokenizer.texts_to_sequences(dfFinal['frases'])
X = pad_sequences(sequences, maxlen=6)

# 4. Prepare labels
y = np.array(dfFinal['Fake'])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# instanciando o modelo

model=Sequential()
model.add(Embedding(input_dim=5000,output_dim=64,input_length=100,mask_zero=True))
model.add(LSTM(64,return_sequences=False,use_cudnn=False,kernel_regularizer=l2(0.001)))
#model.add(Dense(64,activation='relu',kernel_regularizer=l1(0.001)))
#model.add(Dropout(0.4))  # Dropout after first Dense
model.add(Dense(32,activation='relu',kernel_regularizer=l1(0.001)))
model.add(Dropout(0.4))  # Dropout after first Dense
model.add(Dense(16,activation='relu',kernel_regularizer=l1(0.001)))
model.add(Dropout(0.4))  # Dropout after second Dense
model.add(Dense(1,activation='sigmoid'))


# Otimizadores e função-perda
#Mudança na leaning rate 

model.compile(optimizer=Adam(learning_rate=0.0012),
              loss='binary_crossentropy',
              metrics=['accuracy','Precision','Recall','AUC'])


#6. Define early stopping
early_stop = EarlyStopping(
    monitor='val_loss',      # or 'val_accuracy'
    patience=2,              # number of epochs with no improvement before stopping
    restore_best_weights=True)

# %%
# Processamento e treinamento do modelo
# Parâmetros utilizados foram valores que maximizaram os resulados
# 7. Train
model.fit(X_train, y_train,
          epochs=8,
          batch_size=64,
          validation_data=(X_test, y_test),
          callbacks=[early_stop])
# %%
# Resultados do treinamento do modelo:
# 8. Evaluate
loss, accuracy,Precision,Recall,AUC = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")
print(f"Test Loss: {loss:.2f}")
print(f"Test Precision: {Precision:.2f}")
print(f"Test Recall: {Recall:.2f}")
print(f"Test AUC: {AUC:.2f}")

# %%

