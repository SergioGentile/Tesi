# coding=utf-8
"""Il modulo SKLearn contiene tutte gli algoritmi di ML necessari"""
from __future__ import print_function
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="elementwise comparison failed")
warnings.filterwarnings("ignore", message="numpy.core.umath_tests")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")
warnings.filterwarnings("ignore", message="Variables are collinear")
warnings.filterwarnings("ignore", message="Precision and F-score are ill-defined")

import re
from sklearn import cluster, mixture
from sklearn.metrics import confusion_matrix as conf_matr
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from subprocess import call
import os
from sklearn import tree as tr
from sklearn.metrics import classification_report
import numpy as np
import os.path
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering,MeanShift, estimate_bandwidth
np.set_printoptions(threshold=np.nan)



def LabelEncoder(dataset, target, header, risk_level, dataset_name, force_encoding=False):
    """
LabelEncoder effettua il Label Encoding del dataset e del target in ingresso
    :param dataset: il dataset di cui si vuole effettuare il Label Encoding
    :param target: il target di cui si vuole effettuare il Label Encoding
    :param header: header del CSV file
    :param risk_level: il livello per cui si vuole effettuare la classificazione
    :param dataset_name: il nome del dataset
    :param force_encoding: indica se si vuol forzare l'encoding del dataset, non tenendo conto delle codifiche precedenti già salvate sui file della cartella labels
    :return force_encoding: dataset in cui è stato effettuato il label Encoding
    :return target: target in cui è stato effettuato il label Encoding
    :return dataset_encoder: contiene un dictionary [index, label] associato all'encoding
    :return target_encoder: contiene tutte le label del target
    """


    index_dict = 0
    labels = dict()
    column_index = 0
    #Creo i file e la cartella in cui verrà salvato l'encoding.
    #Tale operazione è utile per velocizzare le operazioni di encoding successive.
    directory = './labels/'
    filename_enc = directory + "encoding_" + risk_level + "_" + os.path.basename(dataset_name).split(".csv")[
        0] + "_" + str(len(dataset)) + ".npy"
    filename_dataset = directory + "enc_data_" + risk_level + "_" + os.path.basename(dataset_name).split(".csv")[
        0] + "_" + str(len(dataset)) + ".npy"
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    total_column = float(len(dataset[0]))

    #Capisco quali sono gli indici delle date che dovrò convertire dal formato del CSV al formato yyyy/mm/dd.
    #Tale formato è utile per l'ordinamento.

    date_i_1 = -1
    date_i_2 = -1
    date_i_3 = -1
    date_i_4 = -1
    for i in range(0, len(header)):
        if header[i] == 'DINZI_VLDT_GRZI':
            date_i_1 = i
        if header[i] == 'DFINE_VLDT_GRZI':
            date_i_2 = i
        if header[i] == 'data_prima_immatricolazione':
            date_i_3 = i
        if header[i] == 'data_ultima_voltura':
            date_i_4 = i

    #Se non esistono i file di encoding o forzo l'encoding, sono costretto ad eseguirlo.
    if os.path.exists(filename_enc) == False or os.path.exists(filename_dataset) == False or force_encoding == True:
        #Estraggo tutti i dati colonna per colonna
        for column in dataset.T:
            column_labels = dict()
            print("\rEncoding: " + str(int((column_index / total_column) * 100)) + "%", end="")
            #Salvo in tmp_labels tutti i dati differenti all'interno di una colonna, cosi' da poter associare ad essi un numero
            tmp_labels = []
            for row in column:
                if column_index == date_i_1 or column_index == date_i_2 or column_index == date_i_3 or column_index == date_i_4:
                    row = to_num_date(row)
                if tmp_labels.__contains__(row) == False:
                    tmp_labels.append(row)
            #Ordino i dati. Tale ordinamento è utile per algoritmi come l'albero di decisione,
            #il cui funzionamento è basato su criteri di split del tipo <= e >
            tmp_labels = sorted(tmp_labels)
            #Associo ad ogni label un numero
            for row in tmp_labels:
                column_labels[index_dict] = row
                labels[index_dict] = row
                index_dict = index_dict + 1
            #Converto il dataset
            for i in range(0, len(dataset)):
                if column_index == date_i_1 or column_index == date_i_2 or column_index == date_i_3 or column_index == date_i_4:
                    label = to_num_date(dataset[i, column_index])
                else:
                    label = dataset[i, column_index]
                dataset[i, column_index] = get_key(column_labels, label)
            column_index = column_index + 1
        print("\rEncoding: 100 %")
        #Salvo l'encoding
        dataset = np.array(dataset).astype(np.int)
        np.save(filename_enc, labels)
        np.save(filename_dataset, dataset)
    else:
        labels = np.load(filename_enc).item()
        dataset = np.load(filename_dataset)

    # Eseguo il Label Encoding per il target.
    # In tal caso posso sfruttare il LabelEncoder di SKLearn perchè non ho problemi di ordinamento.
    target_encoder = preprocessing.LabelEncoder()
    target_encoder.fit(target)
    target = target_encoder.transform(target)
    print("\rEncoding: 100%", end="")
    print("\nDone!")
    return dataset, np.array(target).astype(np.int), labels, target_encoder.classes_


def cross_validation(model, model_name, dataset, target, dataset_name, n_fold=2, confusion_matrix=False, statistics=False, labels=None,
                     risk_level=None, bagging=True, max_features=1, max_samples=1, n_estimator=10
                     ):
    """
cross_validation effettua la stratified cross validation sul dataset specificato come parametro.

    :param model: lista dei classificatori da utilizzare
    :param model_name: lista dei nomi associati ai classificatori
    :param dataset: dataset da utilizzare per la cross validation
    :param target: vettore che contiene l'etichetta di classe per ogni entry
	:param dataset_name: nome del dataset che viene utilizzato
    :param n_fold: specifica il numero di fold che verranno utilizzati dalla cross validation
    :param confusion_matrix: bool value che indica se visualizzare o meno a video la matrice di confusione ad ogni iterazione della cross validation
    :param statistics: bool value che indica se visualizzare o meno a video precision, recall, f1-score e support ad ogni iterazione della cross validation
    :param labels: etichette di classe del target
    :param risk_level: livello di rischio per cui si sta effettuando la classificazione
    :param indica se si vuol effettuare la classificazione con l'operatore di bagging
     """

    random_state = 94
    year = dataset_name[len(dataset_name) - 35:len(dataset_name) - 31]
    filename_statistics = "./statistics_" + year + "/statistics_" + year + "_" + risk_level + ".txt"

    #Se il livello di rischio e' cost_caused_claim, trasformo il problema in binario (cost_caused_claim contiene 4 target differenti di default)
    if risk_level == 'cost_caused_claim':
        targetA_012_B_3 = []
        for i in range(0, len(target)):
            if target[i] != 3:
                targetA_012_B_3.append(0)
            else:
                targetA_012_B_3.append(1)
        labels = ['ClassA (range1,2,3)', 'ClassB (range 4)']
        target=np.array(targetA_012_B_3).astype(int)

    #Prendo i modelli (e i suoi nomi) ordinati per recall.
    #Se n_of_clf < 7, verranno utilizzati i classificatori secondo l'ordine specificato.




    print("\n##### CROSS VALIDATION RUN ########\n")

    #Scrivo le informazioni sul run corrente su un file txt che potrà poi essere esaminato.
    with open(filename_statistics, "a") as myfile:
        myfile.write("####### CROSS VALIDATION START #######\n\n")
        myfile.write("############ INFORMATIONS ############\n")
        myfile.write("Cross validation fold: "+str(n_fold)+"\n")
        print("Cross validation fold: " + str(n_fold))
        myfile.write("Algorithm name: " + str(model_name) + "\n")
        myfile.write("Max Features: " + str(max_features) + "\n")
        print("Max Features: " + str(max_features))
        myfile.write("Max Samples: " + str(max_samples) + "\n")
        print("Max Samples: " + str(max_samples))
        myfile.write("Num Estimator: " + str(n_estimator) + "\n")
        print("Algorithm name: " + str(model_name))
        if bagging==True:
            print("Bagging: Yes")
            myfile.write("Bagging: Yes" + "\n")
        else:
            print("Bagging: No")
            myfile.write("Bagging: No" + "\n")
        myfile.write("######################################\n")
        myfile.close()

    start = time.time()
    y_test_tot = np.array([])
    predicted_tot = np.array([])

    #Utilizzo la StratifiedKFold per creare i fold
    kf = StratifiedKFold(n_splits=n_fold, random_state=random_state, shuffle=False).split(dataset, target)
    k = 1
    for train_index, test_index in kf:
        print("####### Fold " + str(k) + "/" + str(n_fold) + " #######")
        progress_clf = 0

        #Separo il training set e il test set
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]

        #Applico l'oversampling.
        #Di default l'oversampling fa in modo di avere un training set con un egual numero di elementi per tutte le classi.

        k = k + 1

        #Applico l'operatore di bagging
        if bagging==True:
            clf = BaggingClassifier(model, max_samples=max_samples, max_features=max_features, n_estimators=n_estimator, bootstrap=True, n_jobs=-1, random_state=random_state)
        else:
            clf = model
        #Appendo il classificatore (su cui si è già fatto il training) a clfs_already_trained.
        #Tale lista sarà utilie in fase di testing.
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)


        acc = accuracy_score(y_test, predicted, normalize=True, sample_weight=None)
        print("\n\nAccuracy on the prediction: ", end="")
        print("%.2f" % acc)

        if confusion_matrix == True:
            cm = conf_matr(y_test, predicted)
            print("\nConfusion matrix...")
            print(pd.DataFrame(cm, index=labels, columns=labels))
        if statistics == True:
            print("\nStatistics...")
            print(classification_report(y_test, predicted, target_names=labels))

        y_test_tot = np.append(y_test_tot, y_test)
        predicted_tot = np.append(predicted_tot, predicted)

    #Stampo tutte le statistiche su di un file
    y_test_tot = np.array(y_test_tot).astype(int)
    predicted_tot = np.array(predicted_tot).astype(int)
    end = time.time()

    with open(filename_statistics, "a") as myfile:
        myfile.write("\nElapsed times for cross validation: " + timer(start, end) + "\n")
        acc = accuracy_score(y_test_tot, predicted_tot, normalize=True, sample_weight=None)
        myfile.write("Accuracy on the prediction: ")
        print("%.2f" % acc, file=myfile)
        print("", file=myfile)
        myfile.write("Confusion Matrix\n")
        cm = conf_matr(y_test_tot, predicted_tot)
        print(pd.DataFrame(cm, index=labels, columns=labels), file=myfile)
        myfile.write("\n")
        print(classification_report(y_test_tot, predicted_tot, target_names=labels), file=myfile)
        myfile.write("\n\n")
        myfile.close()

def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def get_key(labels, label):
    for key, value in labels.items():
        if value == label:
            return key


def tree_as_pdf(dataset, target, features, risk_level, labels, dataset_name=""):
    """
Elabora l'albero di decisione come pdf e lo salva nel path specificato
    :param dataset: dataset su cui eseguire il training
    :param target: target su cui eseguire il training
    :param features: attributi su cui si è basato l'albero di decisione
    :param risk_level: livello di rischio su cui lavora l'albero
    :param labels: etichette del dataset, utili per effettuare il reverse encoding
   """
    #Creo la cartella ./dtree che contiene gli alberi di decisione
    year = dataset_name[len(dataset_name) - 35:len(dataset_name) - 31]
    path = 'dtree_' + year + "_"+ risk_level + '.pdf'  # path dove verra' salvato l'albero di decisione
    if os.path.isdir("./dtree_" + year) == False:
        os.mkdir("./dtree_" + year)
    path = "./dtree_" + year +"/" + path


    print("Save the decision tree as pdf file into " + path)
    #Avvio il classificatore e salvo l'albero come .dot
    tree = DecisionTreeClassifier(max_depth=5).fit(dataset, target)
    tr.export_graphviz(tree,
                       out_file='temp.dot', feature_names=features)

    # Converte i numeri del .dot alle label corrette
    with open('./temp.dot', "r") as f:
        lines = f.readlines()
        f.close()
    with open('./temp.dot', "w") as f:
        for line in lines:
            split_gini = line.split(r"\ngini")
            if len(split_gini) > 1:
                number_gini = split_gini[0].split(" ")
                label = labels[int(float(number_gini[3]))]
                line = line.replace(number_gini[3], label)
            f.write(line + "\n")
        f.close

    call(['dot', '-Tpdf', 'temp.dot', '-o', path])
    print("Done!")
    os.remove('temp.dot')


def to_num_date(date):
    """
    Trasforma una data dal formato ggMMMyyyy nel formato yyyy/mm/dd
    :param date: la data da convertire
"""
    pattern = re.compile("[^0-31][0-31].*")
    if pattern.search(date) is None:
        return date
    months = dict()
    months['JAN'] = 1
    months['FEB'] = 2
    months['MAR'] = 3
    months['APR'] = 4
    months['MAY'] = 5
    months['JUN'] = 6
    months['JUL'] = 7
    months['AUG'] = 8
    months['SEP'] = 9
    months['OCT'] = 10
    months['NOV'] = 11
    months['DEC'] = 12
    month_str = str(date[2:5])
    day_str = date.split(month_str)
    day = day_str[0]
    year = day_str[1]
    conv_date = str(year + "/" + str(months[month_str]).zfill(2) + "/" + day)
    return conv_date
