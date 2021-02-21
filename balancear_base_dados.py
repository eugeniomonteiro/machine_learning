# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:01:15 2020

@author: Eugenio
"""

import pandas as pd
from random import sample
import numpy as np

#Critérios do autor da base
# I ____ Included at abstract or article level

# E ____ Non-specifically excluded

# 1 ____ Excluded due to foreign language

# 2 ____ Excluded due to wrong outcome

# 3 ____ Excluded due to wrong drug

# 4 ____ Excluded due to wrong population

# 5 ____ Excluded due to wrong publication type

# 6 ____ Excluded due to wrong study design

# 7 ____ Excluded due to wrong study duration

# 8 ____ Excluded due to background article

# 9 ____ Excluded due to only abstract being available

# Não considerar 1, 8 e 9


#AtypicalAntiPsychotics
#BetaBlockers
#CalciumChannelBlockers
#ProtonPumpInhibitors

arquivo = 'ProtonPumpInhibitors'

base_orig = pd.read_csv(arquivo + '.csv')

#Selecionar colunas
base_orig = base_orig.loc[:,{'PubMed ID', 'Title', 'Abstract', 'Abstract triage status'}]

#Eliminar linhas com algum campo vazio
base_orig = base_orig.dropna()

#Balancear a base
#Selecionar todos os artigos 'I'(Incluídos)
#Pegar (aleatoriamente) a mesma quantidade de artigos 'E'(Excluídos)
base_inc = base_orig[base_orig['Abstract triage status']=='I']

base_excl_e = base_orig[base_orig['Abstract triage status']=='E']
base_excl_2 = base_orig[base_orig['Abstract triage status']=='2']
base_excl_3 = base_orig[base_orig['Abstract triage status']=='3']
base_excl_4 = base_orig[base_orig['Abstract triage status']=='4']
base_excl_5 = base_orig[base_orig['Abstract triage status']=='5']
base_excl_6 = base_orig[base_orig['Abstract triage status']=='6']
base_excl_7 = base_orig[base_orig['Abstract triage status']=='7']

base_excl = base_excl_e.append(base_excl_2)
base_excl = base_excl.append(base_excl_3)
base_excl = base_excl.append(base_excl_4)
base_excl = base_excl.append(base_excl_5)
base_excl = base_excl.append(base_excl_6)
base_excl = base_excl.append(base_excl_7)


#Cria a coluna que indica se artigo foi incluido (1) ou excluido (0)
base_excl['classe'] = 0
base_inc['classe'] = 1

idx_excl = np.array([i for i in range(len(base_excl))])

for i in range(0,10):
    amostra = sample(list(idx_excl), len(base_inc))
    temp_e = base_excl.iloc[amostra,:]
    temp_i = base_inc
    base = temp_i.append(temp_e)
    base.to_csv(arquivo + '_balanceada_'+str(i)+'.csv')