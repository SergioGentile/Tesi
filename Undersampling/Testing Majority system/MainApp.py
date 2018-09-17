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
import sendEmail
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





for dataset_path in dataset_paths:
    for risk_level in risk_levels:

        print("####### SCRIPT FOR " + str.upper(risk_level) + " START #######\n")

        #print(ReadDataset.read_csv.__doc__)
        #Leggo il CSV e separo header, dataset e target, dove il target e' il livello di rischio di interesse.
        #Dal dataset vengono eliminato gli altri tre livelli di rischio.
        header, dataset, target = ReadDataset.read_csv(path_name=dataset_path, target=risk_level)

        #Trasformo il dataset e il target, cosi da avere solamente valori numerici e poter effettuare la classificazione
        dataset, target, dataset_encoder, target_encoder= SKLearn.LabelEncoder(dataset, target, header, risk_level=risk_level, dataset_name=dataset_path, force_encoding=False )

        #Effettua la leave-one-validation basandosi sul dataset passato come parametro.
        #Nella cartella statistics rilascia le statistiche in formato .txt ed .xls utilizzabili per l'analisi.
        for percentage in range(40, 61, 20):
            for n_clf in range(3,8, 2):
                SKLearn.undersampling_prediction(dataset=dataset,target=target,dataset_name=dataset_path,labels=target_encoder,
                             risk_level=risk_level, percentage_A=percentage, n_clf=n_clf)

        #Trasforma i file txt rilasciati dalla cross validation in excel cosi' da esser letti piu' facilmente.
        txt2xls.txt2xls(path_name=dataset_path)
    year = dataset_path[len(dataset_path) - 35:len(dataset_path) - 31]
    filename_email = 'statistics_' + str(year) + ".xls"
    path_email = "./statistics_" + str(year) + "/"+ filename_email
    sendEmail.send_email(filename_email, path_email)