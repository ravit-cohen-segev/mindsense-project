import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from calc_sumBDI import *

if __name__ == "__main__":
    #load df from before_after file
    before_after = pd.read_excel("before-after.xlsx")
    before_after1 = pd.DataFrame()
    before_after1["c_questionnaire"] = before_after["c_questionnaire"]
    before_after1["heb_code"] = before_after["heb_code"]

    before_after1["SumOQ"] = before_after["SumOQ"]

    #replace all NULLS with np.nan
    before_after2 = before_after1.replace('#NULL!',np.nan)

    #read congruence results file
    cong_res = pd.read_excel("coag_ravit_ex1.xlsx", sheet_name='scores')
    cong_ind = [item[0] for item in pd.read_excel("coag_ravit_ex1.xlsx", sheet_name='indices').values]

    #keep the scores of out test subjects in before_after2 df
    for i,item in enumerate(before_after2['heb_code']):

        # drop the rows that contains only nans in the before questionnaire

        if item == 'כא':

            before_after2 = before_after2.drop([i])

        if item in cong_ind:
            continue
        else:
            before_after2 = before_after2.drop([i])

    before_after2 = before_after2.to_numpy()

    new_before_after_plot = pd.DataFrame(before_after2)
    # split array (for convinience)
    before = before_after2[np.where(before_after2[:, 0] == 1)]
    after = before_after2[np.where(before_after2[:, 0] == 2)]

    #plot before after bar charts for every 10 patients in the list
    width = 0.35  # the width of the bars
    plot_before_after(before, after, 'sumOQ', 'pre-post', before[:,2], after[:,2])





















