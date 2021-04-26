import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Cong_sessions_plots():
    def __init__(self):
        pass
    def plot_reg_line(self, x, l, ax, color):
        i = 0
        while i<l:
            x1 = x[i:i+5,1]
            y1 = x[i:i+5,2]
            z1 = np.polyfit(x1, y1, 1)
            p1 = np.poly1d(z1)
            ax.scatter(x1, y1, c=color)
            ax.plot(x1, p1(x1),  "r--", c=color, label='better')
            i += 5
        return ax

    def create_Xarr(self,x):
        #  add sequential session numbers for every patient
        sess = [1, 2, 3, 4, 5]
        l = len(x)
        add_col = np.array([int(l / 5) * sess])
        x =  np.hstack((x[:, 0].reshape(l, 1), add_col.T, x[:, 1].reshape(l, 1)))
        return x, l

    def plot_groups(self, cong, change, rci):
        worse = change[np.where(change[:,1] < -rci)]
        slight_worse = change[np.where((change[:,1] < 0) & (change[:,1] > -rci)) ]
        better = change[np.where(change[:,1] > rci)]
        slight_better = change[np.where((change[:, 1] > 0) & (change[:, 1] < rci))]
        no_change = change[np.where(change[:, 1] == 0)]
        #plot congruence ratio of all best and worst treatments as a 5 session time series

        pat_worse = np.intersect1d(worse[:,0], cong[:,0])
        p_worse_ind = [np.argwhere(cong[:,0] == pat_worse[i]) for i in range(len(worse))]
        #merge subarrays into one array
        p_worse_ind = np.concatenate(p_worse_ind)
        x_worse = np.concatenate(cong[p_worse_ind,:])

        pat_slight_worse = np.intersect1d(slight_worse[:,0], cong[:,0])
        p_slight_worse_ind = [np.argwhere(cong[:, 0] == pat_slight_worse[i]) for i in range(len(slight_worse))]
        #merge subarrays into one array
        p_slight_worse_ind = np.concatenate(p_slight_worse_ind)
        x_slight_worse = np.concatenate(cong[p_slight_worse_ind,:])

        pat_slight_better = np.intersect1d(slight_better[:,0], cong[:,0])
        p_slight_better_ind = [np.argwhere(cong[:, 0] == pat_slight_better[i]) for i in range(len(slight_better))]
        #merge subarrays into one array
        p_slight_better_ind = np.concatenate(p_slight_better_ind)
        x_slight_better = np.concatenate(cong[p_slight_better_ind,:])

        pat_better = np.intersect1d(better[:,0], cong[:,0])
        p_better_ind = [np.argwhere(cong[:,0] == pat_better[i]) for i in range(len(better))]
        # merge subarrays into one array
        p_better_ind = np.concatenate(p_better_ind)
        x_better = np.concatenate(cong[p_better_ind,:])

        pat_no_change = np.intersect1d(no_change[:,0], cong[:,0])
        p_no_ch_ind = [np.argwhere(cong[:,0] == pat_no_change[i]) for i in range(len(no_change))]
        # merge subarrays into one array
        p_no_ch_ind = np.concatenate(p_no_ch_ind)
        x_no_change = np.concatenate(cong[p_no_ch_ind,:])

        x_worse, l_w = self.create_Xarr(x_worse)
        x_slight_worse, l_s_w =  self.create_Xarr(x_slight_worse)
        x_better, l_b = self.create_Xarr(x_better)
        x_slight_better, l_s_b = self.create_Xarr(x_slight_better)
        x_no_change, l_n_c = self.create_Xarr(x_no_change)

        #plot multiple lines
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        ax1 = self.plot_reg_line(x_better, l_b, ax1, color='green')
        ax1 = self.plot_reg_line(x_worse, l_w, ax1, color='red')
        ax2 = self.plot_reg_line(x_slight_better, l_s_b, ax2, color='green')
        ax2 = self.plot_reg_line(x_slight_worse, l_s_w, ax2, color='red')
        ax2 = self.plot_reg_line(x_no_change, l_n_c, ax2, color='black')

        #number of sessions for each patient
        sess = [1, 2, 3, 4, 5]
        ax1.set_ylabel("congruence/incongruence")
        ax1.set_xticks(sess)
        ax1.legend(['better', 'worse'], loc='upper left')
        ax2.set_ylabel("congruence/incongruence")
        ax2.set_xticks(sess)
        ax2.legend(['slight_better', 'slight_worse', 'no_change'], loc='upper left')

        #manually change colors of legend according to line color
        leg1 = ax1.get_legend()
        leg1.legendHandles[0].set_color('green')
        leg1.legendHandles[1].set_color('red')

        leg2 = ax2.get_legend()
        leg2.legendHandles[0].set_color('green')
        leg2.legendHandles[1].set_color('red')
        leg2.legendHandles[2].set_color('black')
        plt.show()

        return

if __name__ == "__main__":
    #read congruence ratio file and convert to numpy array
    cong_ratio = pd.read_excel("cong_ratio_ex1.xlsx")

    cong_ratio.columns = ['0', 'index', 'cong_ratio']
    #drop this unnecessary index column
    cong_ratio = cong_ratio.drop(['0'], axis=1)
    cong_ratio_arr = cong_ratio.to_numpy()
    change = pd.read_excel("abs_diff_ex1.xlsx")
    change.columns = ['index', 'pre-post']
    change_arr = change.to_numpy()
    g = Cong_sessions_plots()
    g.plot_groups(cong_ratio_arr, change_arr, rci=9.278707271)
