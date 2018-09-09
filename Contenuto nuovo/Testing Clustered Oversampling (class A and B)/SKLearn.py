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
from sklearn.cluster import KMeans
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
    # Creo i file e la cartella in cui verrà salvato l'encoding.
    # Tale operazione è utile per velocizzare le operazioni di encoding successive.
    directory = './labels/'
    filename_enc = directory + "encoding_" + risk_level + "_" + os.path.basename(dataset_name).split(".csv")[
        0] + "_" + str(len(dataset)) + ".npy"
    filename_dataset = directory + "enc_data_" + risk_level + "_" + os.path.basename(dataset_name).split(".csv")[
        0] + "_" + str(len(dataset)) + ".npy"
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    total_column = float(len(dataset[0]))

    # Capisco quali sono gli indici delle date che dovrò convertire dal formato del CSV al formato yyyy/mm/dd.
    # Tale formato è utile per l'ordinamento.

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

    # Se non esistono i file di encoding o forzo l'encoding, sono costretto ad eseguirlo.
    if os.path.exists(filename_enc) == False or os.path.exists(filename_dataset) == False or force_encoding == True:
        # Estraggo tutti i dati colonna per colonna
        for column in dataset.T:
            column_labels = dict()
            print("\rEncoding: " + str(int((column_index / total_column) * 100)) + "%", end="")
            # Salvo in tmp_labels tutti i dati differenti all'interno di una colonna, cosi' da poter associare ad essi un numero
            tmp_labels = []
            for row in column:
                if column_index == date_i_1 or column_index == date_i_2 or column_index == date_i_3 or column_index == date_i_4:
                    row = to_num_date(row)
                if tmp_labels.__contains__(row) == False:
                    tmp_labels.append(row)
            # Ordino i dati. Tale ordinamento è utile per algoritmi come l'albero di decisione,
            # il cui funzionamento è basato su criteri di split del tipo <= e >
            tmp_labels = sorted(tmp_labels)
            # Associo ad ogni label un numero
            for row in tmp_labels:
                column_labels[index_dict] = row
                labels[index_dict] = row
                index_dict = index_dict + 1
            # Converto il dataset
            for i in range(0, len(dataset)):
                if column_index == date_i_1 or column_index == date_i_2 or column_index == date_i_3 or column_index == date_i_4:
                    label = to_num_date(dataset[i, column_index])
                else:
                    label = dataset[i, column_index]
                dataset[i, column_index] = get_key(column_labels, label)
            column_index = column_index + 1
        print("\rEncoding: 100 %")
        # Salvo l'encoding
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
                     risk_level=None, percentage=-1, n_cluster=5, cluster_both_class=True):
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
    random_state = 94
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

    # Prendo i modelli (e i suoi nomi) ordinati per il parametro passato (r,p,f,m)
    models, clfs_name, baggings, max_features, max_samples = get_clf_ordered(risk_level, "r")

    print("\n##### CROSS VALIDATION RUN ########")
    if percentage == -1:
        print("Run oversampling without percentage\n")
    else:
        print("Run oversampling with percentage for the less populous class: " + str(percentage) + "%")
    print("Number of cluster: " + str(n_cluster)+"\n")
    start = time.time()

    #Inizializzo le liste che conterranno i risultati
    y_test_tot = dict()
    predicted_tot = dict()
    index_dict = 0
    for n_of_clf in range(2, len(clfs_name) + 1):
        for n_right in range(2, n_of_clf + 1):
            y_test_tot[index_dict] = np.array([])
            predicted_tot[index_dict] = np.array([])
            index_dict = index_dict + 1

    #Normalizzo il dataset per effettuare il KMeans


    if cluster_both_class == True:
        dataset_discretized = preprocessing.normalize(dataset, norm='l2')
        cl = KMeans(n_clusters=n_cluster).fit(dataset_discretized)
        clusters = cl.labels_
    else:
        dataset_B =[]
        target_B = []

        for i_t, t in enumerate(target):
            if t==1:
                dataset_B.append(dataset[i_t])
                target_B.append(t)


        dataset_discretized_B = preprocessing.normalize(dataset_B, norm='l2')
        cl = KMeans(n_clusters=n_cluster).fit(dataset_discretized_B)
        clusters_B = cl.labels_

        clusters = []
        find_c = 0
        for t in target:
            if t == 0:
                clusters.append(-1)
            else:
                clusters.append(clusters_B[find_c])
                find_c = find_c+1
        clusters = np.array(clusters).astype(int)

    # Utilizzo la StratifiedKFold per creare i fold
    kf = StratifiedKFold(n_splits=n_fold, random_state=random_state, shuffle=False).split(dataset, target)
    k = 1
    for train_index, test_index in kf:
        print("####### Fold " + str(k) + "/" + str(n_fold) + " #######")

        # Separo il training set e il test set
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]
        #Mi prendo solamente le label ti cluster di mio interesse (quelle che stanno nel training)
        clusters_train = clusters[train_index]

        # Applico l'oversampling.
        # Di default l'oversampling fa in modo di avere un training set con un egual numero di elementi per tutte le classi.
        if percentage == -1:
            X_train, y_train = ADASYN().fit_sample(X_train, y_train)
        else:
            # E' possibile (settando il parametro percentage) avere un numero di elementi della classe meno popolosa pari ad una percentuale del dataset iniziale.
            # In entrambi i casi gli elementi su cui verrà fatto oversampling sono scelti in modo randomico.

            # Come si prende una percentuale di elementi in modo randomico?
            # Selezioni gli indici della classe meno popolosa
            count_in_train = 0
            for i in range(0, len(y_train)):
                if y_train[i] == 1:
                    count_in_train = count_in_train + 1

            # Calcolo cB_tot, ovvero in numero di elementi che dovrò avere per raggiungere la percentuale di oversampling specificata
            cB_perc = float(percentage)
            tot = float(len(y_train))
            cB_tot = int((cB_perc / 100.0) * tot) - count_in_train

            # cB_tot mi rappredsenta il numero di elementi che voglio ancora

            #Conto quanti elementi ho per ogni label di clustering
            counter_cluster_train = []
            for i_c in range(0, n_cluster, 1):
                counter_cluster_train.append(0)
            #capisco quindi com'è fatta la popolazione
            for i_t, cluster_target in enumerate(clusters_train):
                if y_train[i_t] == 1:
                    counter_cluster_train[cluster_target] = counter_cluster_train[cluster_target] + 1

            # Capisco quanti dati devono essere aggiunti per ogni target al fine di rispecchiare la percentuale
            cB_tot_x_cluster = int(cB_tot / 5.0)

            #Calcolo quanti elementi devono essere aggiunti per ogni label di cluster per il target 1
            to_add_cluster_counter = []
            for i_c in range(0, n_cluster, 1):
                to_append = 0
                if cB_tot_x_cluster - counter_cluster_train[i_c] > 0:
                    to_append = cB_tot_x_cluster - counter_cluster_train[i_c]
                to_add_cluster_counter.append(to_append)

            #Aggiungo gli elementi
            for i_c, cB_tot_cluster in enumerate(to_add_cluster_counter):
                r_index_B = []
                cB_tot_cluster_added = 0
                #Prendo elementi finchè non raggiungo cB_tot_cluster, cioè la quota da aggiungere desiderata
                while cB_tot_cluster_added < cB_tot_cluster:
                    random_index = random.randint(0, len(clusters_train) - 1)
                    #Prendo solamente gli indici che sono della mia etichetta di cluster e con target 1, non sono interessato logicamente a fare oversampling sul target 0
                    if clusters_train[random_index] == i_c and y_train[random_index] == 1:
                        r_index_B.append(random_index)
                        cB_tot_cluster_added = cB_tot_cluster_added + 1


                # Aggancio questi elementi al training set
                X_train = np.array(list(X_train) + list(X_train[r_index_B])).astype(int)
                y_train = np.array(list(y_train) + list(y_train[r_index_B])).astype(int)

        k = k + 1
        clfs_already_trained = []
        progress_clf = 0
        # Eseguo il training con i vari classificatori
        for model, name, bagging, max_feature, max_sample in zip(models, clfs_name, baggings, max_features, max_samples):
            print("\rTraining progress: " + str(int((float(progress_clf) / float(len(clfs_name))) * 100)) + "%", end="")
            # Applico l'operatore di bagging
            if bagging==True:
                clf = BaggingClassifier(model, n_estimators=7, max_samples=max_sample, max_features=max_feature, random_state=random_state)
            else:
                clf = model
            # Appendo il classificatore (su cui si è già fatto il training) a clfs_already_trained.
            # Tale lista sarà utilie in fase di testing.
            clfs_already_trained.append(clf.fit(X_train, y_train))
            progress_clf = progress_clf + 1
        print("\rTraining progress: " + str(int((float(progress_clf) / float(len(clfs_name))) * 100)) + "%", end="")
        print("\n")
        all_predicted_clf = []
        for clf in clfs_already_trained:
            all_predicted_clf.append(clf.predict(X_test))

        # Effettuo tutte le predizioni utilizzando il test set.
        # Tali predizioni verranno effettuate su tutte le possibili conbinazioni di n_right e n_of_clf e verranno
        # salvate all'interno del file excel al fine di permetterne la visione
        index_dict = 0
        for n_of_clf in range(2, len(clfs_name) + 1):
            predicted_clf = all_predicted_clf[:n_of_clf]
            for n_right in range(2, n_of_clf + 1):
                predicted = []
                for i in range(0, len(predicted_clf[0])):
                    counter_right = 0
                    for j in range(0, n_of_clf):
                        if predicted_clf[j][i] == 1:
                            counter_right = counter_right + 1
                    if (counter_right >= n_right):
                        predicted.append(1)
                    else:
                        predicted.append(0)
                y_test_tot[index_dict] = np.append(y_test_tot[index_dict], y_test)
                predicted_tot[index_dict] = np.append(predicted_tot[index_dict], predicted)
                index_dict = index_dict + 1

    # Stampo tutte le statistiche su di un file
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
                myfile.write("Number of cluster: " + str(n_cluster) + "\n")
                if cluster_both_class == True:
                    myfile.write("Classes used for clustering: " + str("A, B") + "\n")
                else:
                    myfile.write("Classes used for clustering: " + str("A") + "\n")
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
            index_dict = index_dict + 1


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


def get_clf_ordered(risk_level, order):
    """
    A partire da un livello di rischio, torna la configurazione migliore basandosi sulla recall
    :param risk_level: livello di rischio di interesse
    """

    ######## n_caused_claim ########
    classifiers_n_caused_claim = [
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=5),
    ]

    names_n_caused_claim = [
        "QDA",
        "Naive Bayes",
        "AdaBoost",
        "Decision Tree",
    ]

    bagging_n_caused_claim = [
        True,
        True,
        True,
        True,
    ]

    #Ordinati per recall
    max_features_n_caused_claim = [
        0.55,
        0.775,
        1.0,
        1.0,
    ]

    max_samples_n_caused_claim = [
        0.325,
        0.55,
        0.325,
        0.1,
    ]
    ######################

    ######## NNC ########
    classifiers_NNC = [
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=5),
    ]

    names_NNC = [
        "QDA",
        "Naive Bayes",
        "AdaBoost",
        "Decision Tree",
    ]

    bagging_NNC = [
        True,
        True,
        True,
        True,
    ]

    # Ordinati per recall
    max_features_NNC = [
        0.55,
        0.325,
        1.0,
        1.0,
    ]

    max_samples_NNC = [
        1.0,
        0.55,
        1.0,
        0.325,
    ]

    ######################

    ######## n_caused_claim ########
    classifiers_NCD = [
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        AdaBoostClassifier(),
    ]

    names_NCD = [
        "QDA",
        "Naive Bayes",
        "Decision Tree",
        "AdaBoost",
    ]

    bagging_NCD = [
        True,
        True,
        True,
        True,
    ]

    # Ordinati per recall
    max_features_NCD = [
        0.55,
        0.325,
        1.0,
        1.0,
    ]

    max_samples_NCD = [
        0.325,
        0.325,
        0.1,
        1.0,
    ]

    ######################

    ######## n_caused_claim ########
    classifiers_cost_caused_claim = [
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        AdaBoostClassifier(),
    ]

    names_cost_caused_claim = [
        "QDA",
        "Naive Bayes",
        "Decision Tree",
        "AdaBoost",
    ]

    bagging_cost_caused_claim = [
        True,
        True,
        False,
        True,
    ]

    # Ordinati per recall
    max_features_cost_caused_claim = [
        0.55,
        0.325,
        0.0,
        1.0,
    ]

    max_samples_cost_caused_claim = [
        0.775,
        0.55,
        0.0,
        1.0,
    ]

    ######################


    if risk_level == 'cost_caused_claim':
        return classifiers_cost_caused_claim, names_cost_caused_claim, bagging_cost_caused_claim, max_features_cost_caused_claim, max_samples_cost_caused_claim
    if risk_level == 'n_caused_claim':
        return classifiers_n_caused_claim, names_n_caused_claim, bagging_n_caused_claim, max_features_n_caused_claim, max_samples_n_caused_claim
    if risk_level == 'NCD':
        return classifiers_NCD, names_NCD, bagging_NCD, max_features_NCD, max_samples_NCD
    if risk_level == 'NNC':
        return classifiers_NNC, names_NNC, bagging_NNC, max_features_NNC, max_samples_NNC