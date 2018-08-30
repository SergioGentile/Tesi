"""Il modulo ReadDataset.py fornisce la funzione read_csv() che legge il dataset fornitogli come parametro"""

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")



import csv
import numpy as np
import os
import datetime



def read_csv(path_name, head=False, target=''):
    """
read_csv legge il dataset specificato come parametro e separa header, dataset per il training e il target
    :param path_name: path da cui leggere il dataset
    :param head: bool value che specifica se ritornare o meno l'header del dataset (False by default)
    :param target: specifica il livello di rischio su cui si vuole lavorare (uno tra NNC, NCD, n_caused_claim, cost_caused_claim)
    :return header: l'header del csv da cui sono stati rimossi i livelli di rischio
    :return dataset: il dataset del csv da cui sono stati rimossi i livelli di rischio
    :return target: il livello di rischio d'interesse
        """
    if target!='NNC' and target!='NCD' and target!='n_caused_claim' and target!='cost_caused_claim':
        raise Exception('Please choose one of the following argument for \'target\':\nNNC\nNCD\nn_caused_claim\ncost_caused_claim')

    if os.path.isdir("./statistics") == False:
        os.mkdir("./statistics")

    now = datetime.datetime.now()
    with open("./statistics/statistics_" + target + ".txt", "a") as myfile:
        myfile.write("\nSCRIPT RUN "+ now.strftime("%Y-%m-%d %H:%M") + " \n\n")
        myfile.close()

    print('Read the dataset...')
    with open(path_name, 'r') as textfile:
        dataset = list(csv.reader(textfile))
        textfile.close()
    print('Parse the dataset...')
    #Rimuovo l'header
    header = dataset.pop(0)

    #Capisco gli indici degli attributi da rimuovere
    index = 0
    for column in header:
        if column == 'NNC':
            NNC_index = index
        if column == 'NCD':
            NCD_index = index
        if column == 'n_caused_claim':
            n_caused_claim_index = index
        if column == 'cost_caused_claim':
            cost_caused_claim_index = index
        if column == 'NPLZA':
            NPLZA_index = index
        if column == '':
            empty_index = index
        index = index+1

    to_delete = []
    to_delete.append(NPLZA_index)
    to_delete.append(NNC_index)
    to_delete.append(NCD_index)
    to_delete.append(n_caused_claim_index)
    to_delete.append(cost_caused_claim_index)
    to_delete.append(empty_index)

    #Seleziono il target
    if target == 'NNC':
        target = NNC_index
    if target == 'NCD':
        target = NCD_index
    if target == 'n_caused_claim':
        target = n_caused_claim_index
    if target == 'cost_caused_claim':
        target = cost_caused_claim_index

    print('Get the target...')
    #Creo un vettore targets contenente i valori del livello di rischio selezionato per ogni riga
    targets = []
    for row in dataset:
        targets.append(row[target])

    dataset = np.array(dataset)
    dataset = np.delete(dataset, to_delete, 1)

    del header[empty_index]
    del header[cost_caused_claim_index]
    del header[n_caused_claim_index]
    del header[NCD_index]
    del header[NNC_index]
    del header[NPLZA_index]
    header = np.array(header)

    print('Read operation done!')


    if head == True:
        return header, dataset, targets
    return dataset, targets









