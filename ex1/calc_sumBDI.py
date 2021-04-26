import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_before_after_change_ratio(b, a, l): #l is the number of total questiond
    p_change = []
    diff_res = []
    #add 1 to each question in the questionnaire. This is to avoid subsequent division by zero
    to_add = np.ones(len(b))*l
    a += to_add
    b += to_add
    for i in range(len(b)):
        diff = a[i] - b[i]
        diff_res.append(diff)
        ch = diff / b[i]
        p_change.append(ch)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    width =0.35
    x1 = np.arange(len(b))
    rects1 = ax1.bar(x1 - width / 2, p_change, width, label='%change')
    ax1.set_ylabel("%change")
    ax1.set_title("%change from baseline-all patients")
    ax1.set_xticks(x1)

    rects2 = ax2.bar(x1 - width / 2, diff_res, width, label='diff')
    ax2.axhline(y=9.278707271, color='r', linestyle='-')
    ax2.axhline(y=-9.278707271, color='g', linestyle='-')
    ax2.set_ylabel("before-after")
    ax2.set_title("abs change pre_treatment_sumBDI - post_treatment_sumBDI")
    ax2.set_xticks(x1)
    plt.show()

    #convert diff_res to dataframe for saving in excel file
    diff_res = pd.DataFrame(diff_res)
    return diff_res, p_change

def plot_before_after(bef,aft, y_title, graph_title,  annot_b, annot_a):

    for i, val in enumerate(bef[:,1]):
        fig, ax = plt.subplots()
        width = 0.05
        x1 = np.arange(1)
   #    rects1 = ax.bar(x1 - width / 2, bef[i,-1], width, label='before')
    #    rects2 = ax.bar(x1 + width / 2, aft[i,-1], width, label='after')
        diff = bef[i,-1] - aft[i,-1]
        rects3 = ax.bar(x1 + width / 2, diff, width, label='pre-post')
        ax.set_ylabel(y_title)
        ax.set_title(graph_title+' patient '+val[::-1])
        #write values in graph
        txt = "beofre:"+str(annot_b[i])+'\n'+'after:'+str(annot_a[i])
        ax.text(0, -40, txt)

        ax.set_xticks(x1)
        plt.yticks(np.arange(-50,50,10))
        plt.show()
        plt.savefig(r'C:/Users/ravit/PycharmProject/mindsense/BDI_OQ_change_per_patient/' + y_title + val)
    return

if __name__ == "__main__":
    #load df from before_after file
    before_after = pd.read_excel("before-after.xlsx")
    before_after1 = pd.DataFrame()
    before_after1["c_questionnaire"] = before_after["c_questionnaire"]
    before_after1["heb_code"] = before_after["heb_code"]

 #   for i in range(1,22):
  #      before_after1["BDI"+str(i)] = before_after["BDI"+str(i)]
    before_after1["SumBDI"] = before_after["SumBDI"]

    #replace all NULLS with np.nan
    before_after2 = before_after1.replace('#NULL!',np.nan)

    #read congruence results file
    cong_res = pd.read_excel("coag_ravit_ex1.xlsx", sheet_name='scores')
    cong_ind = [item[0] for item in pd.read_excel("coag_ravit_ex1.xlsx", sheet_name='indices').values]

    #keep the scores of out test subjects in before_after2 df
    for i,item in enumerate(before_after2['heb_code']):

        # drop the rows that contains only nans in the before questionnaire

        if item in (['כא','סט']):

            before_after2 = before_after2.drop([i])

        if item in cong_ind:
            continue
        else:
            before_after2 = before_after2.drop([i])

    before_after2['compare_sum'] = np.nansum(before_after2.to_numpy()[:, 2:-1], axis=1)
    # plot before and after multiple time series

    before_after2 = before_after2.to_numpy()

    new_before_after_plot = pd.DataFrame(before_after2)
    # split array (for convinience)
    before = before_after2[np.where(before_after2[:, 0] == 1)]
    after = before_after2[np.where(before_after2[:, 0] == 2)]

    #plot before after bar charts for every 10 patients in the list
    width = 0.35  # the width of the bars
    i = 0

    #plot the last 11 patients
    plot_before_after(before[:,:3],after[:,:3], 'delta-sumBDI', 'pre-post', before[:,2], after[:,2])


    #calculate and plot change ratio for all patients
 #   length_of_ques = 21
 #   abs_diff, change_ratio = plot_before_after_change_ratio(before[:, 23], after[:, 23], length_of_ques)

    #save abs_diff to file. This is for using it later when dividing treatments to groups according to outcomes
 #   abs_diff.to_excel("abs_diff_ex1.xlsx")



















