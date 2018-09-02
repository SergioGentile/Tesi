# coding=utf-8
"""Il modulo SKLearn contiene tutte gli algoritmi di ML necessari"""
from __future__ import print_function
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="elementwise comparison failed")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")
warnings.filterwarnings("ignore", message="Variables are collinear")
warnings.filterwarnings("ignore", message="Precision and F-score are ill-defined")


import re
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

np.set_printoptions(threshold=np.nan)
import time


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



def cross_validation(dataset, target, dataset_name, n_fold=2, labels=None,
                     risk_level=None, percentage=-1):
    """
cross_validation effettua la stratified cross validation sul dataset specificato come parametro.
Essa contiene al suo interno 7 algoritmi che vengono combinati al fine di aumentare le prestazioni.
Di default utilizza la miglior configurazione possibile per ogni livello di rischio, ma è possibile utilizzare anche altre configurazioni.
    :param dataset: dataset da utilizzare per la cross validation
    :param target: vettore che contiene l'etichetta di classe per ogni entry
    :param dataset_name: nome del dataset che viene utilizzato
    :param n_fold: specifica il numero di fold che verranno utilizzati dalla cross validation
    :param labels: etichette di classe del target
    :param risk_level: livello di rischio per cui si sta effettuando la classificazione
    :param percentage: indica la percentuale di elementi ripetuti nell'oversampling rispetto alla lunghezza del dataset iniziale. Di default viene effettuato oversampling andando ad utilizzare un egual numero di entry per tutte le classi.
    """

    year = dataset_name[len(dataset_name) - 35:len(dataset_name) - 31]
    filename_statistics = "./statistics_" + year + "/statistics_" + year + "_" + risk_level + ".txt"

    # Se il livello di rischio e' cost_caused_claim, trasformo il problema in binario (cost_caused_claim contiene 4 target differenti di default)
    if risk_level == 'cost_caused_claim':
        targetA_012_B_3 = []
        for i in range(0, len(target)):
            if target[i] != 3:
                targetA_012_B_3.append(0)
            else:
                targetA_012_B_3.append(1)
        labels = ['ClassA (range1,2,3)', 'ClassB (range 4)']
        target = np.array(targetA_012_B_3).astype(int)

    # Prendo i modelli (e i suoi nomi) ordinati per recall.
    models, clfs_name = get_clf_ordered(risk_level)


    print("\n##### CROSS VALIDATION RUN ########")
    if percentage == -1:
        print("Run oversampling without percentage\n")
    else:
        print("Run oversampling with percentage for the less populous class: " + str(percentage) + "%\n")

    start = time.time()
    y_test_tot = dict()
    predicted_tot = dict()

    index_dict=0
    for n_of_clf in range(2, len(clfs_name) + 1):
        for n_right in range(2, n_of_clf + 1):
            y_test_tot[index_dict] = np.array([])
            predicted_tot[index_dict] = np.array([])
            index_dict=index_dict+1

    # Utilizzo la StratifiedKFold per creare i fold
    kf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False).split(dataset, target)
    k = 1
    for train_index, test_index in kf:
        print("####### Fold " + str(k) + "/" + str(n_fold) + " #######")

        # Separo il training set e il test set
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]

        # Applico l'oversampling.
        # Di default l'oversampling fa in modo di avere un training set con un egual numero di elementi per tutte le classi.
        if percentage == -1:
            X_train, y_train = ADASYN().fit_sample(X_train, y_train)
        else:
            # E' possibile (settando il parametro percentage) avere un numero di elementi della classe meno popolosa pari ad una percentuale del dataset iniziale.
            # In entrambi i casi gli elementi su cui verrà fatto oversampling sono scelti in modo randomico.

            # Come si prende una percentuale di elementi in modo randomico?
            #Selezioni gli indici della classe meno popolosa
            index_B = []
            target_max = max(y_train)
            for i in range(0, len(y_train)):
                if y_train[i] == target_max:
                    index_B.append(i)

            # Conto quanti sono gli elementi della classe popolosa attualmente presenti nel training set
            count_in_train = 0
            for i in range(0, len(y_train)):
                if y_train[i] == target_max:
                    count_in_train = count_in_train + 1
            
            # Calcolo cB_tot, ovvero in numero di elementi che dovrò avere per raggiungere la percentuale di oversampling specificata
            cB_perc = float(percentage)
            tot = float(len(y_train))
            cB_tot = int((cB_perc / 100.0) * tot) - count_in_train
            
            # Creo una lista di numeri random pari a cb_tot
            r_index_B = list()
            for i in range(0, cB_tot):
                r_index_B.append(random.randint(0, len(index_B) - 1))

            # Prendo dalla lista di indici un certo numero di indici (pari alla percentuale) in modo random, sfruttando il vettore r_index_B
            r_indexes_B = []
            for i in r_index_B:
                r_indexes_B.append(index_B[i])
             # Aggancio questi elementi al training set
            X_train = list(X_train) + list(X_train[r_indexes_B])
            y_train = list(y_train) + list(y_train[r_indexes_B])


        k = k + 1
        clfs_already_trained = []
        progress_clf = 0
        #Eseguo il training con i vari classificatori
        for model, name in zip(models, clfs_name):
            print("\rTraining progress: " + str(int((float(progress_clf) / float(7)) * 100)) + "%", end="")
            # Applico l'operatore di bagging
            clf = BaggingClassifier(model)
            # Appendo il classificatore (su cui si è già fatto il training) a clfs_already_trained.
            # Tale lista sarà utilie in fase di testing.
            clfs_already_trained.append(clf.fit(X_train, y_train))
            progress_clf=progress_clf+1
        print("\rTraining progress: " + str(int((float(progress_clf) / float(7)) * 100)) + "%", end="")
        print("\n")
        all_predicted_clf = []
        for clf in clfs_already_trained:
            all_predicted_clf.append(clf.predict(X_test))

        # Effettuo tutte le predizioni utilizzando il test set.
        #Tali predizioni verranno effettuate su tutte le possibili conbinazioni di n_right e n_of_clf e verranno
        #salvate all'interno del file excel al fine di permetterne la visione
        index_dict = 0
        for n_of_clf in range(2, len(clfs_name)+1):
            predicted_clf = all_predicted_clf[:n_of_clf]
            for n_right in range(2, n_of_clf+1):
                predicted = []
                for i in range(0, len(predicted_clf[0])):
                    counter_right = 0
                    for j in range(0, n_of_clf):
                        if predicted_clf[j][i] == 1:
                            counter_right = counter_right+1
                    if(counter_right>= n_right):
                        predicted.append(1)
                    else:
                        predicted.append(0)
                y_test_tot[index_dict] = np.append(y_test_tot[index_dict], y_test)
                predicted_tot[index_dict] = np.append(predicted_tot[index_dict], predicted)
                index_dict = index_dict+1

    #Stampo tutte le statistiche su di un file
    end = time.time()
    index_dict = 0
    for n_of_clf in range(2, len(clfs_name) + 1):
        for n_right in range(2, n_of_clf + 1):
            y_test_tot_iter = y_test_tot[index_dict].astype(int)
            predicted_tot_iter = predicted_tot[index_dict].astype(int)

            with open(filename_statistics, "a") as myfile:
                myfile.write("####### CROSS VALIDATION START #######\n\n")
                myfile.write("############ INFORMATIONS ############\n")
                myfile.write("Cross validation fold: " + str(n_fold) + "\n")
                if percentage == -1:
                    myfile.write("Run oversampling without percentage\n")
                else:
                    myfile.write(
                        "Run oversampling with percentage for the less populous class: " + str(percentage) + "%\n")

                myfile.write(
                    "Number of equal prediction between classification for less populous class: " + str(n_right) + "\n")
                myfile.write("Number of classifier used: " + str(n_of_clf) + "\n")
                myfile.write("######################################\n")
                myfile.write("\nElapsed times for cross validation: " + timer(start, end) + "\n")
                acc = accuracy_score(y_test_tot_iter, predicted_tot_iter, normalize=True, sample_weight=None)
                myfile.write("Accuracy on the prediction: ")
                print("%.2f" % acc, file=myfile)
                print("", file=myfile)
                myfile.write("Confusion Matrix\n")
                cm = conf_matr(y_test_tot_iter, predicted_tot_iter)
                print(pd.DataFrame(cm, index=labels, columns=labels), file=myfile)
                myfile.write("\n")
                print(classification_report(y_test_tot_iter, predicted_tot_iter, target_names=labels), file=myfile)
                myfile.write("\n\n")
                myfile.close()
            index_dict = index_dict+1


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def get_key(labels, label):
    for key, value in labels.items():
        if value == label:
            return key


def tree_as_pdf(dataset, target, features, path, labels):
    """
Elabora l'albero di decisione come pdf e lo salva nel path specificato
    :param dataset: dataset su cui eseguire il training
    :param target: target su cui eseguire il training
    :param features: attributi su cui si è basato l'albero di decisione
    :param path: dove salvare il file
   """

    if os.path.isdir("./dtree") == False:
        os.mkdir("./dtree")
    path = "./dtree/" + path

    print("Save the decision tree as pdf file into " + path)
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
    # input: formato 01JAN1990
    pattern = re.compile("[^0-31][0-31].*")
    if pattern.search(date) is None:
        return date
    months = dict()
    months['JAN'] = 1;
    months['FEB'] = 2;
    months['MAR'] = 3;
    months['APR'] = 4;
    months['MAY'] = 5;
    months['JUN'] = 6;
    months['JUL'] = 7;
    months['AUG'] = 8;
    months['SEP'] = 9;
    months['OCT'] = 10;
    months['NOV'] = 11;
    months['DEC'] = 12;
    month_str = str(date[2:5])
    day_str = date.split(month_str)
    day = day_str[0]
    year = day_str[1]
    conv_date = str(year + "/" + str(months[month_str]).zfill(2) + "/" + day)
    return conv_date

def get_clf_ordered(risk_level):
    """
    A partire da un livello di rischio, torna la configurazione migliore basandosi sulla recall
    :param risk_level: livello di rischio di interesse
    """
    classifiers_NCD = [
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        KNeighborsClassifier()
    ]

    names_NCD = [
        "Naive Bayes",  # 0.79
        "Decision Tree",  # 0.76
        "Neural Net",  # 0.75
        "AdaBoost",  # 0.48
        "QDA",  # 0.46
        "Random Forest",  # 0.24
        "Nearest Neighbor"  # 0.18
    ]

    classifiers_n_caused_claim = [
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        KNeighborsClassifier()
    ]

    names_n_caused_claim = [
        "Naive Bayes",  # 0.79
        "Decision Tree",  # 0.74
        "Neural Net",  # 0.49
        "AdaBoost",  # 0.48
        "QDA",  # 0.43
        "Random Forest",  # 0.28,
        "Nearest Neighbor"  # 0.19
    ]

    classifiers_cost_caused_claim = [
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        QuadraticDiscriminantAnalysis(),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        KNeighborsClassifier()
    ]

    names_cost_caused_claim = [
        "Naive Bayes",  # 0.69
        "QDA",  # 0.71
        "Decision Tree",  # 0.62
        "Neural Net",  # 0.46
        "AdaBoost",  # 0.24,
        "Random Forest",  # 0.26,
        "Nearest Neighbor",  # 0.12
    ]

    classifiers_NNC = [
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        MLPClassifier(alpha=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        KNeighborsClassifier()
    ]

    names_NNC = [
        "QDA",  # 0.81
        "Naive Bayes",  # 0.64
        "Neural Net",  # 0.62
        "Decision Tree",  # 0.57
        "Random Forest",  # 0.22
        "AdaBoost",  # 0.13
        "Nearest Neighbor"  # 0.12
    ]

    if risk_level == 'cost_caused_claim':
        return classifiers_cost_caused_claim, names_cost_caused_claim
    if risk_level == 'n_caused_claim':
        return classifiers_n_caused_claim, names_n_caused_claim
    if risk_level == 'NCD':
        return classifiers_NCD, names_NCD
    if risk_level == 'NNC':
        return classifiers_NNC, names_NNC