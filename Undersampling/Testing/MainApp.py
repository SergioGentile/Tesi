from __future__ import print_function
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")
warnings.filterwarnings("ignore", message="numpy.core.umath_tests")
import SKLearn
import ReadDataset
import txt2xls
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC



######## SETTARE QUI I PARAMETRI #########
dataset_paths = [
    'dataset/itarea_compl2016_telematics_sent_clr_disc_y.csv',
    'dataset/itarea_compl2015_telematics_sent_clr_disc_y.csv'
    ] #path da dove leggere il csv di Reale Mutua

risk_levels = [ #Livelli di rischio da utilizzare
            'n_caused_claim',
            'NCD',
            'cost_caused_claim',
            'NNC',
            ]
##########################################


clfs = [
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        RandomForestClassifier(max_depth=9, min_samples_leaf=1, min_samples_split=4, class_weight='balanced'),
        SVC(C=1, decision_function_shape="ovo", gamma=0.001, coef0=1.0),
        MLPClassifier(),
        KNeighborsClassifier(),
    ]

clfs_name = [
        "Naive Bayes",
        "QDA",
        "AdaBoost",
        "Decision Tree",
        "Random Forest",
        "SVC",
        "Neural Network",
        "KNN"
    ]


for dataset_path in dataset_paths:
    for risk_level in risk_levels:
        for clf, clf_name in zip(clfs, clfs_name):
            print("####### SCRIPT FOR " + str.upper(risk_level) + " START #######\n")

            #print(ReadDataset.read_csv.__doc__)
            #Leggo il CSV e separo header, dataset e target, dove il target e' il livello di rischio di interesse.
            #Dal dataset vengono eliminato gli altri tre livelli di rischio.
            header, dataset, target = ReadDataset.read_csv(path_name=dataset_path, target=risk_level)

            #Trasformo il dataset e il target, cosi da avere solamente valori numerici e poter effettuare la classificazione
            dataset, target, dataset_encoder, target_encoder= SKLearn.LabelEncoder(dataset, target, header, risk_level=risk_level, dataset_name=dataset_path, force_encoding=False )

            #Effettua la leave-one-validation basandosi sul dataset passato come parametro.
            #Nella cartella statistics rilascia le statistiche in formato .txt ed .xls utilizzabili per l'analisi.
            for percentage in range(40, 61, 10):
                SKLearn.undersampling_prediction(model=clf, clf_name = clf_name,dataset=dataset,target=target,dataset_name=dataset_path,labels=target_encoder,
                                     risk_level=risk_level, percentage_A=percentage)

                #Trasforma i file txt rilasciati dalla cross validation in excel cosi' da esser letti piu' facilmente.
                txt2xls.txt2xls(path_name=dataset_path)
