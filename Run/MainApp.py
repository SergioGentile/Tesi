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

######## SETTARE QUI I PARAMETRI #########
dataset_paths = [
    'dataset/itarea_compl2016_telematics_sent_clr_disc_y.csv',
    'dataset/itarea_compl2015_telematics_sent_clr_disc_y.csv'] #path da dove leggere il csv di Reale Mutua

n_fold=5 #Numero di fold utilizzati nella cross validation
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

        #Stampa (formato pdf) l'albero di decisione utilizzato nella classificazione del livello di rischio passato come parametro
        SKLearn.tree_as_pdf(dataset=dataset, target=target, features=header, risk_level=risk_level, labels=dataset_encoder, dataset_name=dataset_path)

        #Effettua la cross validation basandosi sul dataset passato come parametro.
        #Nella cartella statistics rilascia le statistiche in formato .txt utilizzabili per l'analisi.
        SKLearn.cross_validation(dataset=dataset,
                                 target=target,
                                 dataset_name=dataset_path,
                                 labels=target_encoder,
                                 n_fold=n_fold,
                                 confusion_matrix=False,
                                 statistics=False,
                                 risk_level=risk_level
                                 )
    #Trasforma i file txt rilasciati dalla cross validation in excel cosi' da esser letti piu' facilmente.
    txt2xls.txt2xls(path_name=dataset_path)
