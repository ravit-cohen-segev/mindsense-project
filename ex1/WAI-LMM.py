import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import imageio
from sklearn.preprocessing import MinMaxScaler

#normalize target
def NormalizeTaget(y):
    y_scaled = np.tanh(y.astype('float32'))
    return y_scaled

#for removing letters from string
def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

#split data df to train and test
def split_data(data, n, ratio=0.75):
    pat_num = np.arange(n)
    #permute indices
    pat_num = np.random.permutation(pat_num)
    pos_holder = int(ratio*n)
    train_idx = pat_num[:pos_holder]
    test_idx = pat_num[pos_holder:]
    train = data.iloc[data.index[np.in1d(data['group_idx'],train_idx)]]
    test = data.iloc[data.index[np.in1d(data['group_idx'],test_idx)]]
    return train, test

def remove_outliers(data, n):
    new_data = np.empty((0,3))
    removed = np.empty((0,2))
    for i in range(n):
        small_data = data.loc[data['group_idx'] == float(i)]
        # calculate interquartile range
        q25, q75 = np.percentile(small_data.Weight, 25), np.percentile(small_data.Weight, 75)
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        # identify outliers
        outliers = [x for x in small_data.Weight if x < lower or x > upper]
        if outliers != []:
            removed = np.vstack((removed, [i, outliers]))
        # remove outliers
        outliers_removed = small_data[small_data.Weight>=lower]
        outliers_removed = outliers_removed[outliers_removed.Weight <= upper]

        new_data = np.vstack((new_data, outliers_removed))
    new_data = pd.DataFrame(new_data)
    new_data.columns = data.columns
    return new_data, removed

def plot_results(data, uniq, data_name, path): #plot train or test results
    #get number of patients in data
    uniq_idx = np.unique(data['group_idx'])
    figs = []

    for i in uniq_idx:
        #create small df containing all of the observations for one patient
        plot_data = data.loc[data['group_idx'] == float(i)]
        plot_data['x_points']  = np.arange(len(plot_data))
        plt.plot(plot_data['x_points'],  plot_data['pred_results'], marker='o', color='r')
        plt.plot(plot_data['x_points'], plot_data['True_results'], marker='o', color='b')
        plt.ylabel('WAI')
        plt.xlabel('n_observations')
        plt.legend(['predicted-WAI', 'actual-WAI'])
        plt.title(data_name +'_results' + str(uniq[int(i)]))

        #save to file and read with imageio tp prepare a clip
        filename = path + data_name +'_results' + str(uniq[int(i)])
        plt.savefig(filename)
        figs.append(imageio.imread(filename +'.png'))
        plt.close()
    imageio.mimsave(data_name +'.gif', figs, duration=0.8)

    return

if __name__=='__main__':
    np.random.seed(30)
    #read sbs file
    sbs_df = pd.read_excel('sbs.xlsx', skiprows=1)
    #reset column names
    sbs_df = sbs_df[1:].rename(columns=sbs_df.iloc[0])
    #save only 3 columns name, session number, and WAI measurement
    WAI_df = pd.DataFrame()
    WAI_df['heb_code'] = sbs_df['heb_code']
    WAI_df['session_n'] = sbs_df['session_n']
    WAI_df['c_a_wai'] = sbs_df['c_a_wai']
    #delete sbs_df
    del sbs_df

    #upload congruence results with indices and session number
    cong_df = pd.read_excel('coag_ravit_ex1.xlsx', names=['sess_n', 'congruence'], sheet_name='scores')

    #read the names from indices sheet in the same file
    cong_df['heb_name'] = pd.read_excel('coag_ravit_ex1.xlsx', header=None, sheet_name='indices')

    #keep patients for subsequent filtering
    filter_pat = pd.DataFrame()
    filter_pat['heb_name'] = cong_df['heb_name']

    #set patients names as indices
    cong_df = cong_df.set_index(cong_df['heb_name'])

    #remove column
    cong_df = cong_df.drop(['heb_name'], axis=1)

    #strip letters from meetings, keep only session numbers
    cong_df['sess_n'] = [only_numerics(val) for val in cong_df['sess_n']]

    #reformeat cong_df to np.array, columns will be sessions and rows will be patient name (one row for each)
    x_inp = np.empty((2,0))
    #get all of the indices
    ind2 = cong_df.index.unique().values
    for idx in ind2:
        #get all values with same index and place them in one row
        cong_values = cong_df.iloc[cong_df.index == idx]['congruence'].values
        to_append = np.vstack(([idx]*len(cong_values), cong_values))
        x_inp = np.hstack((x_inp, to_append))

    #transpose x_inp matrix
    x_inp = x_inp.T
    # keep session numbers for subsequent filtering
    filter_pat['sess_n'] = cong_df['sess_n'].values

    #convert WAI to numpy for easier processing
    WAI_arr = WAI_df.to_numpy()

    #keep only patients with codes
    new_arr = np.empty((0,3))
    WAI_ch_arr = np.empty((0,2))

    for i, row in enumerate(WAI_arr):
        #get WAI only from patients with congruence sessions
        if row[0] in list(filter_pat['heb_name']):
            ind_li = filter_pat.loc[(filter_pat['heb_name'] == row[0])]
            if str(row[1]) in list(ind_li['sess_n']):
                prev = WAI_arr[i-1]
                #if WAI is NULL don't add the rows
                if row[2]=='#NULL!' or prev[2]=='#NULL!':
                    continue
                new_arr = np.vstack([new_arr, row])
                new_arr = np.vstack([new_arr, prev])
                #calculate change in WAI between the following and current sessions
                WAI_ch = row[2] - prev[2]
                WAI_ch_arr = np.vstack((WAI_ch_arr, (row[0], WAI_ch)))

    #get unsorted unique values of patient names
    uniq, index = np.unique(WAI_ch_arr[:, 0], return_index=True)
    uniq = uniq[index.argsort()]

    y_output = np.empty((2,0))
    for u in uniq:
        row_WAI = WAI_ch_arr[np.where(WAI_ch_arr[:,0] == u), 1][0].astype('float')
        #if there are NANS replace with median value
        if (np.isnan(row_WAI)).any():
            #compute median without NAN
            med = np.nanmedian(row_WAI)
            #replace nans with median
            row_WAI[np.where(np.isnan(row_WAI))] = med

        row_WAI = np.vstack(([u]*len(row_WAI), row_WAI))
        y_output = np.hstack((y_output, row_WAI))

    #transpose y_ouput matrix
    y_output = y_output.T
    # sort by patient names to make sure that indices match the indices in x_inp
    y_output = y_output[y_output[:, 0].argsort()]
    #keep only values without indices
    y_output = y_output[:,1:]

    #plot data to understand the distribution
#    fig = plt.figure()
 #   for i, arr in enumerate(x_inp[:,1:]):
  #      fig = plt.scatter(arr, y_output[i,:].T)
   # plt.show()
   # plt.close()
    #create LMM with random fixed effects

    data = pd.DataFrame()
    data['Time'] = x_inp[:,1]
    data['Weight'] = y_output
    #give numeric values to group indices for LMM model
    group_idx = []
    for i in range(len(uniq)):
        group_idx.extend([i]*5)

    data['group_idx'] = group_idx
    data = data.astype(dtype='float32')

    #normalize data
    #x = data.values  # returns a numpy array
    #min_max_scaler = MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(x)
    #data_scaled = pd.DataFrame(x_scaled)
    #data_scaled.columns = data.columns

    #plot sns boxplot for identifying outliers
    sns_fig = sns.boxplot(x='group_idx', y='Weight', data=data, palette="Set3")

    sns_fig.set(xticks=[], yticks=np.arange(-11,11))

    #save figure to file
    fig = sns_fig.get_figure()
    fig.savefig('boxplot_data.png')

    #remove outliers
    data, removed = remove_outliers(data, len(uniq))

    #split dataset to train and test
    train, test = split_data(data, len(uniq), ratio=0.75)

    model = sm.MixedLM.from_formula("Weight ~ Time", data=train, groups=train['group_idx'])
    result = model.fit()

    train_results = pd.DataFrame()
    test_results = pd.DataFrame()

    train_results['pred_results'] = result.predict(train)
    train_results['True_results'] = train['Weight']
    #add train indices to results
    train_results['group_idx'] = train['group_idx']
    train_results['Time'] = train['Time']

    test_results['pred_results'] = result.predict(test)
    test_results['True_results'] = test['Weight']
    #add test indices to results
    test_results['group_idx'] = test['group_idx']
    test_results['Time'] = test['Time']

    #plt.close previous graphs
    plt.close()
    path = r'C:\Users\ravit\PycharmProject\mindsense\ex1\LMM-files/'
    #plot without outliers
    plot_results(train_results, uniq, 'Train_without_outliers', path=path)
    plot_results(test_results, uniq, 'Test_without_outliers', path=path)

    #plot with outliers
    #plot_results(train_results, uniq, 'Train_with_outliers', path=path)
    #plot_results(test_results, uniq, 'Test_with_outliers', path=path)





















