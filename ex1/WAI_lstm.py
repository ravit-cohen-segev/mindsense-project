import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, Model
from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

#normalize target
def NormalizeTaget(y):
    y_scaled = np.tanh(y.astype('float32'))
    return y_scaled

#for removing letters from string
def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

#perform grid search of optimal hyperparametes for neural networks

class lstm_cong():
    def __init__(self, x, y):
        self.x_shape = (1,5)
        self.x = x.reshape(len(x), 1, len(x.T))
       #self.y = y
        self.y = y.reshape(len(y), 1, len(y.T))

    def create_model(self, loss_fn = 'mse', optimization=Adam(0.001)):
        lstm_units = 32
        latent_dim = 16
        kernel_regularizer = regularizers.l2(1e-5)
        initializer = tf.keras.initializers.RandomNormal()

        lstm_input = Input(shape=self.x_shape, name='input_layer')

        lstm_layer1 = LSTM(lstm_units, kernel_regularizer=kernel_regularizer, kernel_initializer=initializer,
                           return_sequences=True, activation=tf.keras.layers.LeakyReLU(), name='lstm_layer1')(lstm_input)
        batch_layer1 = BatchNormalization()(lstm_layer1)
        dense_layer = Dense(latent_dim, activation=tf.keras.layers.LeakyReLU(), name='Dense_layer')(batch_layer1)
        batch_layer2 = BatchNormalization()(dense_layer)

        lstm_layer2 = LSTM(5, kernel_regularizer=kernel_regularizer, return_sequences=True,
                           activation=tf.keras.layers.LeakyReLU(), name='lstm_layer2')(batch_layer2)

        lstm_model = Model(lstm_input, lstm_layer2, name='lstm_model')

        lstm_model.compile(loss=loss_fn, optimizer=optimization,
                                  metrics=[tf.keras.metrics.MeanSquaredError()])
        return lstm_model


    def train_model(self, batches, epochs):
        #build_model
        lstm_model = self.create_model()
        for epoch in range(epochs):
            print("Begin epoch:"+str(epoch))
            lstm_model.fit(self.x, self.y, batch_size=batches)
            print("END of epoch:" + str(epoch))
        return lstm_model

if __name__=='__main__':
    np.random.seed(30)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
    x_inp = np.empty((0,5))
    #get all of the indices
    ind2 = cong_df.index.unique().values
    for idx in ind2:
        #get all values with same index and place them in one row
        cong_values = cong_df.iloc[cong_df.index == idx]['congruence'].values
        x_inp = np.vstack((x_inp, cong_values))
    x_inp = np.hstack((ind2.reshape(len(ind2),1), x_inp))

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

    y_output = np.empty((0,5))
    for u in uniq:
        row_WAI = WAI_ch_arr[np.where(WAI_ch_arr[:,0] == u), 1][0].astype('float')
        #if there are NANS replace with median value
        if (np.isnan(row_WAI)).any():
            #compute median without NAN
            med = np.nanmedian(row_WAI)
            #replace nans with median
            row_WAI[np.where(np.isnan(row_WAI))] = med

        y_output = np.vstack((y_output, row_WAI))
    y_output = np.hstack((uniq.reshape(len(uniq),1), y_output))

    #sort by patient names to make sure that indices match the indices in x_inp
    y_output = y_output[y_output[:, 0].argsort()]
    y_output = y_output[:,1:]
    #for scaling the output

  #  y_output = NormalizeTaget(y_output[:, 1:])
    #plot data to understand the distribution
    fig = plt.figure()
    for i, arr in enumerate(x_inp[:5,1:]):
        fig = plt.scatter(arr, y_output[i,:].T)
    plt.show()




    X_train, X_test, y_train, y_test = train_test_split(x_inp, y_output, test_size = 0.25, random_state = 42)

    x_train_float = X_train[:,1:].astype('float32')
    y_train_float = y_train[:,1:].astype('float32')

 #   class_lstm = lstm_cong(x_train_float, y_train_float)

  #  lstm_model = class_lstm.train_model(batches=5, epochs=50)

    x_test_float = X_test[:,1:].astype('float32')
   # x_test_float = x_test_float.reshape(len(x_test_float), 1, len(x_test_float.T))
    y_test_float = y_test[:, 1:].astype('float32')
   # y_test_float = y_test_float.reshape(len(x_test_float), 1, len(x_test_float.T))

   # test_output = lstm_model.predict(x_test_float)
    #evaluate_test = lstm_model.evaluate(x_test_float, y_test_float)
    #model_weights = lstm_model.get_weights()


















