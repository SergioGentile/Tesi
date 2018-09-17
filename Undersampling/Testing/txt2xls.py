import xlwt
from os import listdir
from os.path import isfile, join

import os

def txt2xls(path_name):
    c1 = 19
    c2 = 20
    year = path_name[len(path_name) - 35:len(path_name) - 31]
    folder_statistics = "./statistics_" + year + "/"
    filename_xls = "statistics_" + year + ".xls"
    filename_statistics = folder_statistics + filename_xls

    onlyfiles = list([f for f in listdir(folder_statistics) if isfile(join(folder_statistics, f))])
    book = xlwt.Workbook()


    if onlyfiles.__contains__(filename_xls) == True:
        os.remove(filename_statistics)
        onlyfiles.remove(filename_xls)

    for file in onlyfiles:
        if file.__contains__(".txt") == False:
            onlyfiles.remove(file)

    for filename in onlyfiles:
        path = folder_statistics+filename
        header_exel = ["Perc_Undersampling","Accuracy","Prec_T0","Rec_T0","F1_T0","Prec_T1","Rec_T1","F1_T1","AVG_Prec","AVG_Rec","AVG_F1","TP", "TN", "FP", "FN","Run_Time", "Clf_Name"]
        filename_split = filename.split(".txt")
        sheet_name = filename_split[0]

        sh = book.add_sheet(sheet_name[16:])
        style = xlwt.XFStyle()
        font = xlwt.Font()
        font.bold = True
        style.font = font
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['ice_blue']
        style.pattern = pattern
        for i, col in enumerate(header_exel):
            sh.write(0, i, col, style=style)
            if col == "Run_Time":
                sh.col(i).width = len(col)*500
            elif col =='Clf_Name':
                sh.col(i).width = len(col) * 500
            elif col == "TP" or col=="TN" or col == "FP" or col == "FN":
                sh.col(i).width = len(col) * 1000
            else:
                sh.col(i).width = len(col) * 310

        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['yellow']
        style.pattern = pattern

        r1 = 3
        sh.write(r1, c1, "Legenda", style)
        r1 = r1 + 1
        sh.write(r1, c1, "Parametro", style)
        sh.write(r1, c2, "Descrizione:",style)
        r1 = r1 + 1
        sh.write(r1, c1, "Perc_Undersampling:")
        sh.write(r1, c2, "Indica la percentuale di undersampling applicata")
        r1 = r1 + 1
        sh.write(r1, c1, "Accuracy:")
        sh.write(r1, c2, "Indica l'accuratezza")
        r1 = r1 + 1
        sh.write(r1, c1, "Prec_X:")
        sh.write(r1, c2, "Indica la precisione nella predizione del target X")
        r1 = r1 + 1
        sh.write(r1, c1, "Rec_X:")
        sh.write(r1, c2, "Indica il richiamo nella predizione del target X")
        r1 = r1 + 1
        sh.write(r1, c1, "F1_X:")
        sh.write(r1, c2, "Indica l'F1-Score nella predizione del target X")
        r1 = r1 + 1
        sh.write(r1, c1, "AVG_Prec:")
        sh.write(r1, c2, "Indica la media della precisione tra la classe 0 e 1")
        r1 = r1 + 1
        sh.write(r1, c1, "AVG_Rec:")
        sh.write(r1, c2, "Indica la media del richiamo tra la classe 0 e 1")
        r1 = r1 + 1
        sh.write(r1, c1, "AVG_F1:")
        sh.write(r1, c2, "Indica la media dell'F1-Score tra la classe 0 e 1")
        r1 = r1 + 1
        sh.write(r1, c1, "TP:")
        sh.write(r1, c2, "Veri Positivi")
        r1 = r1 + 1
        sh.write(r1, c1, "TN:")
        sh.write(r1, c2, "Veri Negativi")
        r1 = r1 + 1
        sh.write(r1, c1, "FP:")
        sh.write(r1, c2, "Falsi Positivi")
        r1 = r1 + 1
        sh.write(r1, c1, "FN:")
        sh.write(r1, c2, "Falsi Negativi")
        r1 = r1 + 1
        sh.write(r1, c1, "Run_Time:")
        sh.write(r1, c2, "Indica il momento in cui e' stato eseguito lo script per quella classificazione")
        r1 = r1 + 1
        sh.write(r1, c1, "Clf_Name:")
        sh.write(r1, c2, "Classificatore utilizzato")

        sh.col(c1).width = len("Perc_Undersampling")*400
        sh.col(c2).width = 110*270

        with open(path) as f:
            max_f1=-1
            max_perc=-1
            max_row=-1
            lines = list(f.readlines())
            percentage = -1
            find_conf_mtr = 0
            find_stat_mtr = 0
            print_line = False
            index_excel = 0
            clf_name=""
            for line in lines:

                if line.__contains__("Algorithm name: "):
                    l_split = line.split(": ")
                    clf_name = l_split[1]
                if line.__contains__("SCRIPT RUN "):
                    l_split = line.split("SCRIPT RUN ")
                    run_time = l_split[1]
                if line.__contains__("Undersampling percenta"):
                    perc_num_split = line.split(": ")
                    percentage = int(perc_num_split[1])
                if line.__contains__("without per"):
                    percentage=0
                elif line.__contains__("with per"):
                    with_perc = line.split(": ")
                    number = with_perc[1].split("%")
                    percentage = int(number[0])
                if line.__contains__("Accuracy on the"):
                    l_split = line.split(": ")
                    accuracy = float(l_split[1])

                ####### LETTURA MATRICE CONFUSIONE #########
                if filename.__contains__("cost"):
                    if find_conf_mtr == 4:
                        find_conf_mtr = 0
                    if find_conf_mtr == 3:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(4, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x10 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x11 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1

                    if find_conf_mtr == 2:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(4, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x00 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x01 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1
                    if find_conf_mtr == 1:
                        find_conf_mtr=find_conf_mtr+1
                    if line.__contains__("Confus"):
                        find_conf_mtr=find_conf_mtr+1
                else:
                    if find_conf_mtr == 4:
                        find_conf_mtr = 0
                    if find_conf_mtr == 3:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(2, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x10 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x11 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1

                    if find_conf_mtr == 2:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(2, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x00 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x01 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1
                    if find_conf_mtr == 1:
                        find_conf_mtr = find_conf_mtr + 1
                    if line.__contains__("Confus"):
                        find_conf_mtr = find_conf_mtr + 1
                    #############################################

                if filename.__contains__("cost") == True:
                    if find_stat_mtr == 4:
                        print_line = True
                        find_stat_mtr = 0
                    if find_stat_mtr == 3:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(7, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s10 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s11 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1

                    if find_stat_mtr == 2:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(7, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s00 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s01 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1
                    if find_stat_mtr == 1:
                        find_stat_mtr = find_stat_mtr + 1
                    if line.__contains__("precisi"):
                        find_stat_mtr = find_stat_mtr + 1
                else:
                    if find_stat_mtr == 4:
                        print_line = True
                        find_stat_mtr = 0
                    if find_stat_mtr == 3:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(15, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s10 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s11 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1

                    if find_stat_mtr == 2:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(15, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s00 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s01 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1
                    if find_stat_mtr == 1:
                        find_stat_mtr = find_stat_mtr + 1
                    if line.__contains__("precisi"):
                        find_stat_mtr = find_stat_mtr + 1

                if print_line == True:
                    print_line = False
                    #header = "Right index,Num Classifiers,Accuracy,Prec T0,Rec T0,F1 T0,Prec T1,Rec T1,F1 T1"
                    if percentage == 0:
                        percentage_p = "No"
                    else:
                        percentage_p = str(percentage) + "%"
                    if float(s00+s01) > 0.0:
                        f11 = float("{0:.2f}".format(float(s00*s01*2)/float(s00+s01)))
                    else:
                        f11 = 0.0
                    if float(s10+s11) > 0.0:
                        f12 = float("{0:.2f}".format(float(s10*s11*2)/float(s10+s11)))
                    else:
                        f12 = 0.0

                    c3=0
                    index_excel = index_excel + 1
                    sh.write(index_excel, c3, percentage_p)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, accuracy)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, s00)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, s01)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, f11)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, s10)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, s11)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, f12)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, "{0:.2f}".format(100*(s00 + s10)/2.0))
                    c3 = c3 + 1
                    sh.write(index_excel, c3, "{0:.2f}".format(100*(s01 + s11)/2.0))
                    c3 = c3 + 1
                    sh.write(index_excel, c3, "{0:.2f}".format(100*(f11 + f12)/2.0))
                    c3 = c3 + 1
                    sh.write(index_excel, c3, x00)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, x11)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, x01)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, x10)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, run_time)
                    c3 = c3 + 1
                    sh.write(index_excel, c3, clf_name)
                    if float((f12 + f11)/2.0)>float(max_f1):
                        max_f1=float((f12 + f11)/2.0)
                        max_perc=percentage_p
                        max_row = index_excel+1
                        max_clf_name = clf_name


        f.close()
        r1 = r1+3
        sh.write(r1, c1, "Risultato migliore", style=style)
        r1 = r1+1
        sh.write(r1, c1, "Riga " + str(max_row)+ ". Si ha un F1-Score medio="+"{0:.2f}".format(100*(f12 + f11)/2.0)+"%.")
        r1 = r1 + 1
        sh.write(r1, c1, "Prametri utilizzati: percentage=" + str(max_perc) + " e classificatore=" + str(max_clf_name))
        r1 = r1 + 1
        sh.write(r1, c1, "Entry della classe A: " + str(x00 + x01))
        r1 = r1 + 1
        sh.write(r1, c1, "Entry della classe B: " + str(x10 + x11))

        style = xlwt.XFStyle()
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['green']
        style.pattern = pattern
        rows = sh.get_rows()
        style0 = xlwt.easyxf('pattern: pattern solid, fore_colour sea_green')
        styleindex = book.add_style(style0)
        for c in range(0, len(header_exel)):
            rows[max_row - 1]._Row__cells[c].xf_idx = styleindex




    if len(onlyfiles)!= 0:
        book.save(filename_statistics)
