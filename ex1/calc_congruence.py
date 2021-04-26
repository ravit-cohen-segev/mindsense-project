import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#lists of codes
patient = ['PPC', 'PPI', 'PPP', 'PTC', 'PTI', 'PTP', 'PDC', 'PDI', 'PDP']
therapist = ['TPC', 'TPI', 'TPP', 'TTC', 'TTI', 'TTP', 'TDC', 'TDI', 'TDP']
out_of_matrix = ['POM', 'TOM']

class Coaguration():
    def __init__(self, matrix):
        self.matrix = matrix
    # calculate coagurance ratio formula
    def coag_ratio(self, echos, total):
        return echos / total

    def check_match(self, a, b):
        pair_l = [a,b]
        #check if out of matrix. If so, check coagurance
        if all(item in pair_l for item in out_of_matrix):
            if a!=b:
                return 'yes'

        #check if same speaker

        if a[0]==b[0]:
            return 'same_speaker'

        if a in patient and b in therapist:
            a_i = patient.index(a)
            b_i = therapist.index(b)
            if a_i == b_i:
                return 'yes'
            else:
                return None
        else:
            if a in therapist and b in patient:
                a_i = therapist.index(a)
                b_i = patient.index(b)
                if a_i == b_i:
                    return 'yes'
                else:
                    return None
        return

    def count_coag_meetings(self):
        print("new_meeting")
        coag_res = []

        for i, meeting in enumerate(self.matrix):
            # count how tom. Maybe useful for calculations later on
            # how many cong pairs and how many breaks
            count_echos = 0
            count_breaks = 0

            for j, code in enumerate(meeting):

                if j == len(meeting) - 1:
                    break

                if code in ['TN', 'PN', np.nan]:
                    continue

                next = j+1
                following_code = meeting[next]
                # if end of conversation

                while following_code in ['TN', 'PN', np.nan]:
                    next += 1
                    if next == len(meeting)-1:
                        break
                    following_code = meeting[next]

                if next == len(meeting)-1:
                    following_code = meeting[next]
                    if following_code in ['TN', 'PN', np.nan]:
                        break
                #check if code and prev are a coagurance or a break
                check = self.check_match(code, following_code)

                if check == 'same_speaker':
                    continue

                if check == 'yes':
                    count_echos += 1

                if check is None:
                    count_breaks += 1

            count_all = count_echos + count_breaks

            coag_val = self.coag_ratio(count_echos, count_all)
            coag_res.append(coag_val)
        return coag_res
    @staticmethod
    def median_coag_res(coag_res):
        med_coag_res = []
        i = 0
        while i < len(coag_res)-1:
            med_coag_res.append(np.median(coag_res[i:i+5]))
            i += 5
        return med_coag_res

if __name__ == "__main__":
    igum_data = pd.read_excel("igum_codim.xlsx")
    #inverse rows and columns
    igum_data = igum_data.T
    #create new indices:
    new_indices = []
    num_patients = int(len(igum_data) / 5)
    for i in range(num_patients):
        new_indices.extend([i]*5)
    old_indices = igum_data.index
    new_indices = pd.Series(new_indices)
    igum_data_after = igum_data

    igum_data_after = igum_data_after.set_index(new_indices)
    #save file with transposed data
 #   trans_data = pd.DataFrame(igum_data_after)
  #  trans_data = trans_data.set_index(old_indices)
   # trans_data.to_excel("transposed_df.xlsx")
    #switch to numpy array
    igum_data_after = igum_data_after.to_numpy()
    co = Coaguration(igum_data_after)
    cong_res = co.count_coag_meetings()

    #save to file with index with old indices
  #  to_file = pd.DataFrame({'cong_res':cong_res})

  #  to_file = to_file.set_index(old_indices)
#   save to file
   # to_file.to_excel("coag_ravit_ex1.xlsx")
   #plot congruenc/incongruence ratio
    df_cong = pd.DataFrame({'cong_ratio': cong_res})
    df_cong = df_cong.set_index(old_indices)
    med_coag_res = co.median_coag_res(cong_res)

    #for plotting I removed two sessions that I saw in calc_sumBDI.py don't have an after results
    #remove סט כא

    df_cong = df_cong.drop(['כא4','כא8','כא12','כא16','כא18'])
    df_cong = df_cong.drop(['סט3', 'סט9', 'סט13', 'סט20', 'סט25'])
    ind_for_plot_legend = df_cong.index

    new_ind_for_plot = []

    for i in range(int(len(df_cong)/5)):
        new_ind_for_plot.extend([i] * 5)

    new_ind_for_plot = pd.Series(new_ind_for_plot)
    df_cong = df_cong.set_index(new_ind_for_plot)

    #rearrange data for plotting a timeseries. columns are sessions arranged from 1 to 5 left to right
    new_array = np.empty((0,5))
    df_cong.reset_index(inplace=True)

    sess = [1, 2, 3, 4, 5]

    #create an array that contains all of the patients and their meetings
    for i in range(51):
        new_array = np.vstack((new_array, df_cong['cong_ratio'].loc[df_cong['index']==i].values.T))

    #plot a timeseries of congruence for every patient
    for i, arr in enumerate(new_array):
        fig, ax = plt.subplots()
        ax.plot(sess, arr)
        ax.set_xticks(sess)
        ax.set_title('')
        ax.set_ylabel('congruence ratio')
        plt.yticks(np.arange(0.3, 1, 0.1))
        plt.savefig(r'C:/Users/ravit/PycharmProject/mindsense/BDI_OQ_change_per_patient/' + 'congruence' + str(i))
        plt.show()



    '''
    cols = [1,2,3,4,5]
    new_df1 = pd.DataFrame(new_array[:10,:], columns=cols).set_index(np.arange(10)).T
    new_df2 = pd.DataFrame(new_array[10:20,:], columns=cols).set_index(np.arange(10,20)).T
    new_df3 = pd.DataFrame(new_array[20:30,:], columns=cols).set_index(np.arange(20,30)).T
    new_df4 = pd.DataFrame(new_array[30:40, :], columns=cols).set_index(np.arange(30,40)).T
    new_df5 = pd.DataFrame(new_array[40:51, :], columns=cols).set_index(np.arange(40,51)).T

    x_ticks = [1,2,3,4,5,6]
    new_df1.plot(xticks=x_ticks, xlabel='sessions', ylabel='congruence / incongruence', title='congruence_5_sessions')
    plt.legend(loc='best')
    new_df2.plot(xticks=x_ticks, xlabel='sessions', ylabel='congruence / incongruence', title='congruence_5_sessions')
    plt.legend(loc='best')
    new_df3.plot(xticks=x_ticks, xlabel='sessions', ylabel='congruence / incongruence', title='congruence_5_sessions')
    plt.legend(loc='best')
    new_df4.plot(xticks=x_ticks, xlabel='sessions', ylabel='congruence / incongruence', title='congruence_5_sessions')
    plt.legend(loc='best')
    new_df5.plot(xticks=x_ticks, xlabel='sessions', ylabel='congruence / incongruence', title='congruence_5_sessions')
    plt.legend(loc='best')
    plt.show()
    '''''
    #save cong_ratio to file
    df_cong.to_excel("cong_ratio_ex1.xlsx")


