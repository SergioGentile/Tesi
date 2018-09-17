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


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
import re
from sklearn.metrics import confusion_matrix as conf_matr
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
from subprocess import call
import os
from sklearn import tree as tr
from sklearn.metrics import classification_report
import numpy as np
import os.path
import time
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



def undersampling_prediction(dataset, target, dataset_name, labels=None, risk_level=None, percentage_A=50, n_clf=7):

    """
undersampling_prediction applica la tecnica dell'undersampling sul dataset passato come parametro.
Successivamente viene applicata la LeaveOneOut validation per analizzare la classificazione.
    """

    start = time.time()
    year = dataset_name[len(dataset_name) - 35:len(dataset_name) - 31]
    filename_statistics = "./statistics_" + year + "/statistics_" + year + "_" + risk_level + ".txt"

    #Divido il problema in binario per cost_caused_claim
    if risk_level == 'cost_caused_claim':
        targetA_012_B_3 = []
        for i in range(0, len(target)):
            if target[i] != 3:
                targetA_012_B_3.append(0)
            else:
                targetA_012_B_3.append(1)
        labels = ['ClassA (range1,2,3)', 'ClassB (range 4)']
        target=np.array(targetA_012_B_3).astype(int)

    #Capisco quanti elementi devo selezionare dalla classe A per effettuare l'undersampling
    print("Perform undersampling (Percentage = " + str(percentage_A) +")")
    tot_B = count_class_B(target)
    percentage_B = 100-percentage_A
    tot_AB =  int(tot_B*100/float(percentage_B))
    tot_A = int(tot_AB*percentage_A/100.0)
    #print("Target A: from " + str(count_class_A(target)) + " to " + str(tot_A))
    #print("Target A: from " + str(count_class_B(target)) + " to " + str(tot_B))

    #Ottengo le posizioni degli elementi di A e B in modo da poterli campionare
    index_A = []
    index_B = []
    for i,t in enumerate(target):
        if t == 0:
            index_A.append(i)
        else:
            index_B.append(i)


    #Campiono casualmente gli elementi di A
    index_A_selected = []
    for i in random.sample(range(0, len(index_A)), tot_A):
        index_A_selected.append(index_A[i])
    indexes_selected = np.array(index_A_selected + index_B).astype(int)

    dataset = dataset[indexes_selected]
    target = target[indexes_selected]



    tot_progress = len(dataset)
    progress = 0
    #Effettuo la LeaveOneOut validation
    loo = LeaveOneOut()
    loo.get_n_splits(dataset)
    target_predicted = []
    first_time = True
    print("\rValidation progress: 0%", end="")

    for train_index, test_index in loo.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]
        if first_time==True:
            start_clf = time.time()

        clfs_trained = []
        #Training dei classificatori
        clfs, clfs_name = get_clf_ordered(risk_level)
        pos = 1
        for clf, clf_name in zip(clfs, clfs_name):
            if pos>n_clf:
                break
            pos=pos+1
            clf.fit(X_train, y_train)
            clfs_trained.append(clf)
        point_A = 0
        point_B = 0
        for clf in clfs_trained:
            if clf.predict(X_test) == 0:
                point_A = point_A+1
            else:
                point_B = point_B+1
        """print("**************")
        print("Point_A: " + str(point_A))
        print("Point_B: " + str(point_B))"""
        if point_A>point_B:
            prediction=0
        else:
            prediction=1
        """print("Prediction: " + str(prediction))
        print("Right result: " + str(y_test))
        print("**************")"""
        #inserisco il risultato della classificazione in target_predicted, che alla fine conterrà la lista di tutti gli elementi predetti
        target_predicted.append(prediction)
        if first_time == True:
            first_time=False
            end_clf = time.time()
        #Tempo rimanente per finire la validation (utile per printare il tempo sulla console)
        time_miss = (tot_progress-progress)*(end_clf - start_clf)
        progress = progress + 1
        print("\rValidation progress: " + str(int(progress/float(tot_progress)*100)) + "%\tTime miss: " + timer_miss(time_miss), end="")
    print("\rValidation progress: 100%")

    end = time.time()
    # Scrivo le informazioni sul run corrente su un file txt che potrà poi essere esaminato.
    with open(filename_statistics, "a") as myfile:
        myfile.write("####### VALIDATION RUN #######\n\n")
        myfile.write("############ INFORMATIONS ############\n")
        myfile.write("Number classifiers: " + str(n_clf) + "\n")
        print("Number classifiers: " + str(n_clf))
        myfile.write("Undersampling percentage: " + str(percentage_A)+"\n")
        print("Undersampling percentage: " + str(percentage_A))
        myfile.write("######################################\n")
        myfile.write("\nElapsed times for cross validation: " + timer(start, end) + "\n")
        acc = accuracy_score(target, target_predicted, normalize=True, sample_weight=None)
        myfile.write("Accuracy on the prediction: ")
        print("%.2f" % acc, file=myfile)
        print("", file=myfile)
        myfile.write("Confusion Matrix\n")
        cm = conf_matr(target, target_predicted)
        print(pd.DataFrame(cm, index=labels, columns=labels), file=myfile)
        print(pd.DataFrame(cm, index=labels, columns=labels), file=myfile)
        myfile.write("\n")
        cr = classification_report(target, target_predicted, target_names=labels)
        print(cr, file=myfile)
        print(cr)
        myfile.write("\n\n")
        myfile.close()


def count_class_A(target):
    count = 0
    for t in target:
        if t == 0:
            count = count + 1
    return count

def count_class_B(target):
    count = 0
    for t in target:
        if t == 1:
            count = count + 1
    return count


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def timer_miss(time):
    hours, rem = divmod(time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


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


def get_clf_ordered(risk_level):
    """
    A partire da un livello di rischio, torna la configurazione migliore basandosi sulla recall
    :param risk_level: livello di rischio di interesse
    """


    ######## n_caused_claim ########
    classifiers_n_caused_claim = [
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        RandomForestClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        GaussianNB(),
        KNeighborsClassifier(),
        MLPClassifier(),
        QuadraticDiscriminantAnalysis(),

    ]

    names_n_caused_claim = [
        "AdaBoost",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "KNN",
        "Neural Network",
        "QDA",
    ]


    ######################

    ######## NNC ########
    classifiers_NNC = [
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        RandomForestClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        KNeighborsClassifier(),
        MLPClassifier(),

    ]

    names_NNC = [
        "AdaBoost",
        "Decision Tree",
        "Random Forest",
        "QDA",
        "Naive Bayes",
        "KNN",
        "Neural Network",
    ]


    ######################

    ######## NCD ########
    classifiers_NCD = [
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        RandomForestClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(),
        MLPClassifier(),
    ]

    names_NCD = [
        "AdaBoost",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "QDA",
        "KNN",
        "Neural Network",
    ]


    ######################

    ######## cost_caused_claim ########
    classifiers_cost_caused_claim = [
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        RandomForestClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        GaussianNB(),
        KNeighborsClassifier(),
        MLPClassifier(),
        QuadraticDiscriminantAnalysis(),
    ]

    names_cost_caused_claim = [
        "AdaBoost",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "KNN",
        "Neural Network",
        "QDA",
    ]

    ######################


    if risk_level == 'cost_caused_claim':
        return classifiers_cost_caused_claim, names_cost_caused_claim
    elif risk_level == 'n_caused_claim':
        return classifiers_n_caused_claim, names_n_caused_claim
    elif risk_level == 'NCD':
        return classifiers_NCD, names_NCD
    elif risk_level == 'NNC':
        return classifiers_NNC, names_NNC
    else:
        return classifiers_n_caused_claim, names_n_caused_claim


