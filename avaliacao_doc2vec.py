# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:39:47 2020

@author: Eugenio
"""


import pandas as pd
import numpy as np

import time

import nltk
nltk.download('stopwords')



from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier




from keras.models import Sequential
from keras.layers import Dense

#-----------------------------------------------------------------------------
def arredondar(vetor):
    
    vet_a = []
    for i in range(0, len(vetor)):
        if vetor[i] <= 0.5:
            vet_a.append(0)
        if vetor[i] > 0.5:
            vet_a.append(1)
    
    return np.array(vet_a, int)

#------------------------------------------------------------------------------
def criar_corpus(texto):

    documents = []

    for i in range(0, len(texto)):
        doc = texto.loc[i,'Title'] + ' ' + texto.loc[i,'Abstract']
  
        documents.append(doc)
            
    return documents
#------------------------------------------------------------------------------

def preprocessar_texto(data):
    
    corpus = criar_corpus(data)
    
    lista = []
    
    for i in range(0, len(corpus)):
        
        doc_sem_stopwords = remove_stopwords(corpus[i])
       
        doc_preprocessado = simple_preprocess(doc_sem_stopwords)
        

        pmid = data.loc[i,'PubMed ID']
        lista.append(TaggedDocument(words=doc_preprocessado, tags=[str(pmid)]))
        
    return lista

#-----------------------------------------------------------------------------
    




#========================== CÓDIGO PRINCIPAL ===========================================
tamVetor = 200

df_resultados = pd.DataFrame(index=['Atyp_0','Atyp_1','Atyp_2','Atyp_3','Atyp_4','Atyp_5',
                                         'Atyp_6','Atyp_7','Atyp_8','Atyp_9',
                                         'Beta_0','Beta_1','Beta_2','Beta_3','Beta_4','Beta_5',
                                         'Beta_6','Beta_7','Beta_8','Beta_9',
                                         'Calcium_0','Calcium_1','Calcium_2','Calcium_3','Calcium_4',
                                         'Calcium_5','Calcium_6','Calcium_7','Calcium_8','Calcium_9',
                                         'Proton_0','Proton_1','Proton_2','Proton_3','Proton_4',
                                         'Proton_5','Proton_6','Proton_7','Proton_8','Proton_9'],
                                  columns=['LogisticRegression','SVM','DecisionTree','RandomForest', 'DeepLearning'])


df_saidas_modelos = pd.DataFrame(index=['Atyp_0','Atyp_1','Atyp_2','Atyp_3','Atyp_4','Atyp_5',
                                         'Atyp_6','Atyp_7','Atyp_8','Atyp_9',
                                         'Beta_0','Beta_1','Beta_2','Beta_3','Beta_4','Beta_5',
                                         'Beta_6','Beta_7','Beta_8','Beta_9',
                                         'Calcium_0','Calcium_1','Calcium_2','Calcium_3','Calcium_4',
                                         'Calcium_5','Calcium_6','Calcium_7','Calcium_8','Calcium_9',
                                         'Proton_0','Proton_1','Proton_2','Proton_3','Proton_4',
                                         'Proton_5','Proton_6','Proton_7','Proton_8','Proton_9'],
                                  columns=['LogisticRegression','SVM','DecisionTree','RandomForest', 'DeepLearning'])

df_saidas_reais = pd.DataFrame(index=['Atyp_0','Atyp_1','Atyp_2','Atyp_3','Atyp_4','Atyp_5',
                                         'Atyp_6','Atyp_7','Atyp_8','Atyp_9',
                                         'Beta_0','Beta_1','Beta_2','Beta_3','Beta_4','Beta_5',
                                         'Beta_6','Beta_7','Beta_8','Beta_9',
                                         'Calcium_0','Calcium_1','Calcium_2','Calcium_3','Calcium_4',
                                         'Calcium_5','Calcium_6','Calcium_7','Calcium_8','Calcium_9',
                                         'Proton_0','Proton_1','Proton_2','Proton_3','Proton_4',
                                         'Proton_5','Proton_6','Proton_7','Proton_8','Proton_9'],
                                  columns=['Saida_Real'])


for i in range(0,10):
   inicio = time.time()
   print('================== Etapa: '+str(i)+'\n')
   data_atyp = pd.read_csv('dataset/cohen/AtypicalAntipsychotics_balanceada_'+str(i)+'.csv')
   data_beta = pd.read_csv('dataset/cohen/BetaBlockers_balanceada_'+str(i)+'.csv')
   data_calcium = pd.read_csv('dataset/cohen/CalciumChannelBlockers_balanceada_'+str(i)+'.csv')
   data_proton = pd.read_csv('dataset/cohen/ProtonPumpInhibitors_balanceada_'+str(i)+'.csv')
   
   #------------- Variáveis dependentes e independentes -------------------
   x_atyp = data_atyp.loc[:,{'Title','Abstract','PubMed ID'}]
   y_atyp = data_atyp.loc[:,{'classe'}]
   y_atyp = np.ravel(y_atyp)

   x_beta = data_beta.loc[:,{'Title','Abstract','PubMed ID'}]
   y_beta = data_beta.loc[:,{'classe'}]
   y_beta = np.ravel(y_beta)
   
   x_calcium = data_calcium.loc[:,{'Title','Abstract','PubMed ID'}]
   y_calcium = data_calcium.loc[:,{'classe'}]
   y_calcium = np.ravel(y_calcium)
   
   x_proton = data_proton.loc[:,{'Title','Abstract','PubMed ID'}]
   y_proton = data_proton.loc[:,{'classe'}]
   y_proton = np.ravel(y_proton)

   #----------- Preprocessar os textos ------------------------------------
   lista_atyp = preprocessar_texto(x_atyp)
   lista_beta = preprocessar_texto(x_beta)
   lista_calcium = preprocessar_texto(x_calcium)
   lista_proton = preprocessar_texto(x_proton)


   #--------- Doc2Vec -----------------------------------------------------
   model_atyp = Doc2Vec(window=3, dm=1, workers=2, vector_size=tamVetor, min_count=5, epochs=200)
   model_beta = Doc2Vec(window=3, dm=1, workers=2, vector_size=tamVetor, min_count=5, epochs=200)
   model_calcium = Doc2Vec(window=3, dm=1, workers=2, vector_size=tamVetor, min_count=5, epochs=200)
   model_proton = Doc2Vec(window=3, dm=1, workers=2, vector_size=tamVetor, min_count=5, epochs=200)


   model_atyp.build_vocab(lista_atyp)
   model_beta.build_vocab(lista_beta)
   model_calcium.build_vocab(lista_calcium)
   model_proton.build_vocab(lista_proton)
   
   
   model_atyp.train(lista_atyp, total_examples=model_atyp.corpus_count, epochs=model_atyp.epochs)
   model_beta.train(lista_beta, total_examples=model_beta.corpus_count, epochs=model_beta.epochs)
   model_calcium.train(lista_calcium, total_examples=model_calcium.corpus_count, epochs=model_calcium.epochs)
   model_proton.train(lista_proton, total_examples=model_proton.corpus_count, epochs=model_proton.epochs)
   
   #------------- Gerar o vetor de características de cada artigo ---------

   features_atyp = np.zeros((len(lista_atyp),tamVetor))
   for j in range(0,len(lista_atyp)):
       texto = lista_atyp[j][0]
       v = model_atyp.infer_vector(texto)
       for k in range(0,tamVetor):
           features_atyp[j,k] = v[k]
       
   features_beta = np.zeros((len(lista_beta),tamVetor))
   for j in range(0,len(lista_beta)):
       texto = lista_beta[j][0]
       v = model_beta.infer_vector(texto)
       for k in range(0,tamVetor):
           features_beta[j,k] = v[k]          
           
   features_calcium = np.zeros((len(lista_calcium),tamVetor))
   for j in range(0,len(lista_calcium)):
       texto = lista_atyp[j][0]
       v = model_calcium.infer_vector(texto)
       for k in range(0,tamVetor):
           features_calcium[j,k] = v[k]           
           
   features_proton = np.zeros((len(lista_proton),tamVetor))
   for j in range(0,len(lista_proton)):
       texto = lista_proton[j][0]
       v = model_proton.infer_vector(texto)
       for k in range(0,tamVetor):
           features_proton[j,k] = v[k]

   #------------ Separar treino de teste ----------------------------------
   x_trein_atyp, x_test_atyp, y_trein_atyp, y_test_atyp = train_test_split(features_atyp, y_atyp, test_size=0.20, random_state=0)
   x_trein_beta, x_test_beta, y_trein_beta, y_test_beta = train_test_split(features_beta, y_beta, test_size=0.20, random_state=0)
   x_trein_calcium, x_test_calcium, y_trein_calcium, y_test_calcium = train_test_split(features_calcium, y_calcium, test_size=0.20, random_state=0)
   x_trein_proton, x_test_proton, y_trein_proton, y_test_proton = train_test_split(features_proton, y_proton, test_size=0.20, random_state=0)


   #====================== MODELOS =========================================
   class_log = LogisticRegression(random_state=0, max_iter=200)
   class_svc = svm.SVC()
   class_tree = tree.DecisionTreeClassifier(random_state = 0)
   class_forest = RandomForestClassifier(max_depth = 15, random_state = 0)

   #----- Atyp
   class_log.fit(x_trein_atyp, y_trein_atyp)
   class_svc.fit(x_trein_atyp, y_trein_atyp)
   class_tree.fit(x_trein_atyp, y_trein_atyp)
   class_forest.fit(x_trein_atyp, y_trein_atyp)

   score_log = class_log.score(x_test_atyp, y_test_atyp)
   score_svc = class_svc.score(x_test_atyp, y_test_atyp)
   score_tree = class_tree.score(x_test_atyp, y_test_atyp)
   score_forest = class_forest.score(x_test_atyp, y_test_atyp)
   
   deep_learning = Sequential()
   
   #Colocar as camadas no modelo
   deep_learning.add(Dense(x_trein_atyp.shape[1], activation='relu', input_dim = x_trein_atyp.shape[1]))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(100, activation='relu'))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(1, activation='sigmoid'))

   deep_learning.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

   deep_learning.fit(x_trein_atyp, y_trein_atyp, epochs=10, batch_size=3, verbose=1)
   
   score_deep = deep_learning.evaluate(x_test_atyp, y_test_atyp,verbose=1)     
   
   #----- saídas reais
   df_saidas_reais.loc['Atyp_'+str(i),'Saida_Real'] = y_test_atyp
      
   df_resultados.loc['Atyp_'+str(i),'LogisticRegression'] = score_log
   df_resultados.loc['Atyp_'+str(i),'SVM'] = score_svc
   df_resultados.loc['Atyp_'+str(i),'DecisionTree'] = score_tree
   df_resultados.loc['Atyp_'+str(i),'RandomForest'] = score_forest
   df_resultados.loc['Atyp_'+str(i),'DeepLearning'] = score_deep
   
   #----- saídas de cada modelo
   df_saidas_modelos.loc['Atyp_'+str(i),'LogisticRegression'] = class_log.predict(x_test_atyp)
   df_saidas_modelos.loc['Atyp_'+str(i),'SVM'] = class_svc.predict(x_test_atyp)
   df_saidas_modelos.loc['Atyp_'+str(i),'DecisionTree'] = class_tree.predict(x_test_atyp)
   df_saidas_modelos.loc['Atyp_'+str(i),'RandomForest'] = class_forest.predict(x_test_atyp)
   df_saidas_modelos.loc['Atyp_'+str(i),'DeepLearning'] = arredondar(deep_learning.predict(x_test_atyp)) 

   #---- Beta
   class_log = LogisticRegression(random_state=0, max_iter=200)
   class_svc = svm.SVC()
   class_tree = tree.DecisionTreeClassifier(random_state = 0)
   class_forest = RandomForestClassifier(max_depth = 15, random_state = 0)
   
   class_log.fit(x_trein_beta, y_trein_beta)
   class_svc.fit(x_trein_beta, y_trein_beta)
   class_tree.fit(x_trein_beta, y_trein_beta)
   class_forest.fit(x_trein_beta, y_trein_beta)  
   
   score_log = class_log.score(x_test_beta, y_test_beta)
   score_svc = class_svc.score(x_test_beta, y_test_beta)
   score_tree = class_tree.score(x_test_beta, y_test_beta)
   score_forest = class_forest.score(x_test_beta, y_test_beta)
 
   deep_learning = Sequential()
   
   #Colocar as camadas no modelo
   deep_learning.add(Dense(x_trein_beta.shape[1], activation='relu', input_dim = x_trein_beta.shape[1]))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(100, activation='relu'))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(1, activation='sigmoid'))

   deep_learning.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

   deep_learning.fit(x_trein_beta, y_trein_beta, epochs=10, batch_size=3, verbose=1)
   
   score_deep = deep_learning.evaluate(x_test_beta, y_test_beta, verbose=1)     

   #----- saídas reais
   df_saidas_reais.loc['Beta_'+str(i),'Saida_Real'] = y_test_beta

   df_resultados.loc['Beta_'+str(i),'LogisticRegression'] = score_log
   df_resultados.loc['Beta_'+str(i),'SVM'] = score_svc
   df_resultados.loc['Beta_'+str(i),'DecisionTree'] = score_tree
   df_resultados.loc['Beta_'+str(i),'RandomForest'] = score_forest
   df_resultados.loc['Beta_'+str(i),'DeepLearning'] = score_deep
   
   
   #----- saídas de cada modelo
   df_saidas_modelos.loc['Beta_'+str(i),'LogisticRegression'] = class_log.predict(x_test_beta)
   df_saidas_modelos.loc['Beta_'+str(i),'SVM'] = class_svc.predict(x_test_beta)
   df_saidas_modelos.loc['Beta_'+str(i),'DecisionTree'] = class_tree.predict(x_test_beta)
   df_saidas_modelos.loc['Beta_'+str(i),'RandomForest'] = class_forest.predict(x_test_beta)
   df_saidas_modelos.loc['Beta_'+str(i),'DeepLearning'] = arredondar(deep_learning.predict(x_test_beta)) 

   #---- Calcium
   class_log = LogisticRegression(random_state=0, max_iter=200)
   class_svc = svm.SVC()
   class_tree = tree.DecisionTreeClassifier(random_state = 0)
   class_forest = RandomForestClassifier(max_depth = 15, random_state = 0)
   
   class_log.fit(x_trein_calcium, y_trein_calcium)
   class_svc.fit(x_trein_calcium, y_trein_calcium)
   class_tree.fit(x_trein_calcium, y_trein_calcium)
   class_forest.fit(x_trein_calcium, y_trein_calcium)

   score_log = class_log.score(x_test_calcium, y_test_calcium)
   score_svc = class_svc.score(x_test_calcium, y_test_calcium)
   score_tree = class_tree.score(x_test_calcium, y_test_calcium)
   score_forest = class_forest.score(x_test_calcium, y_test_calcium)

   deep_learning = Sequential()
   
   #Colocar as camadas no modelo
   deep_learning.add(Dense(x_trein_calcium.shape[1], activation='relu', input_dim = x_trein_calcium.shape[1]))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(100, activation='relu'))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(1, activation='sigmoid'))

   deep_learning.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

   deep_learning.fit(x_trein_calcium, y_trein_calcium, epochs=10, batch_size=3, verbose=1)
   
   score_deep = deep_learning.evaluate(x_test_calcium, y_test_calcium,verbose=1)   

   #----- saídas reais
   df_saidas_reais.loc['Calcium_'+str(i),'Saida_Real'] = y_test_calcium
   
   df_resultados.loc['Calcium_'+str(i),'LogisticRegression'] = score_log
   df_resultados.loc['Calcium_'+str(i),'SVM'] = score_svc
   df_resultados.loc['Calcium_'+str(i),'DecisionTree'] = score_tree
   df_resultados.loc['Calcium_'+str(i),'RandomForest'] = score_forest   
   df_resultados.loc['Calcium_'+str(i),'DeepLearning'] = score_deep
   
   #----- saídas de cada modelo
   df_saidas_modelos.loc['Calcium_'+str(i),'LogisticRegression'] = class_log.predict(x_test_calcium)
   df_saidas_modelos.loc['Calcium_'+str(i),'SVM'] = class_svc.predict(x_test_calcium)
   df_saidas_modelos.loc['Calcium_'+str(i),'DecisionTree'] = class_tree.predict(x_test_calcium)
   df_saidas_modelos.loc['Calcium_'+str(i),'RandomForest'] = class_forest.predict(x_test_calcium)
   df_saidas_modelos.loc['Calcium_'+str(i),'DeepLearning'] = arredondar(deep_learning.predict(x_test_calcium)) 

   #---- Proton
   class_log = LogisticRegression(random_state=0, max_iter=200)
   class_svc = svm.SVC()
   class_tree = tree.DecisionTreeClassifier(random_state = 0)
   class_forest = RandomForestClassifier(max_depth = 15, random_state = 0)
   
   class_log.fit(x_trein_proton, y_trein_proton)
   class_svc.fit(x_trein_proton, y_trein_proton)
   class_tree.fit(x_trein_proton, y_trein_proton)
   class_forest.fit(x_trein_proton, y_trein_proton)

   score_log = class_log.score(x_test_proton, y_test_proton)
   score_svc = class_svc.score(x_test_proton, y_test_proton)
   score_tree = class_tree.score(x_test_proton, y_test_proton)
   score_forest = class_forest.score(x_test_proton, y_test_proton)

   deep_learning = Sequential()
   
   #Colocar as camadas no modelo
   deep_learning.add(Dense(x_trein_proton.shape[1], activation='relu', input_dim = x_trein_proton.shape[1]))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(100, activation='relu'))
   deep_learning.add(Dense(50, activation='relu'))
   deep_learning.add(Dense(1, activation='sigmoid'))

   deep_learning.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

   deep_learning.fit(x_trein_proton, y_trein_proton, epochs=10, batch_size=3, verbose=1)
   
   score_deep = deep_learning.evaluate(x_test_proton, y_test_proton, verbose=1)  

   #----- saídas reais
   df_saidas_reais.loc['Proton_'+str(i),'Saida_Real'] = y_test_proton

   df_resultados.loc['Proton_'+str(i),'LogisticRegression'] = score_log
   df_resultados.loc['Proton_'+str(i),'SVM'] = score_svc
   df_resultados.loc['Proton_'+str(i),'DecisionTree'] = score_tree
   df_resultados.loc['Proton_'+str(i),'RandomForest'] = score_forest
   df_resultados.loc['Proton_'+str(i),'DeepLearning'] = score_deep


   #----- saídas de cada modelo
   df_saidas_modelos.loc['Proton_'+str(i),'LogisticRegression'] = class_log.predict(x_test_proton)
   df_saidas_modelos.loc['Proton_'+str(i),'SVM'] = class_svc.predict(x_test_proton)
   df_saidas_modelos.loc['Proton_'+str(i),'DecisionTree'] = class_tree.predict(x_test_proton)
   df_saidas_modelos.loc['Proton_'+str(i),'RandomForest'] = class_forest.predict(x_test_proton)
   df_saidas_modelos.loc['Proton_'+str(i),'DeepLearning'] = arredondar(deep_learning.predict(x_test_proton)) 

   fim = time.time()
   
   print('================== TEMPO DECORRIDO: '+str(fim-inicio)+'\n')

#salva os resultados em formato Excel
df_saidas_reais.to_excel('saidas_reais_doc2vec_novo.xls')
df_resultados.to_excel('resultados_doc2vec_novo.xls')
df_saidas_modelos.to_excel('saidas_modelos_doc2vec_novo.xls')