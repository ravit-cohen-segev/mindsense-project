import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, Model
from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

#for removing letters from string
def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

class lstm_cong():
    def __init__(self, x, y):
        self.x_shape = (1,5)
        self.x = x.reshape(len(x), 1, len(x.T))
        self.y = y.reshape(len(y), 1, len(y.T))

    def create_model(self, loss_fn = 'mse', optimization=SGD(0.001)):
        lstm_units = 32
        latent_dim = 16
        kernel_regularizer = regularizers.l2(1e-5)
        initializer = tf.keras.initializers.RandomNormal()

        lstm_input = Input(shape=self.x_shape, name='input_layer')
        lstm_layer1 = LSTM(lstm_units, kernel_initializer=initializer, return_sequences=True, activation=tf.keras.layers.LeakyReLU(),
                           name='lstm_layer1')(lstm_input)
        batch_layer1 = BatchNormalization()(lstm_layer1)
        dense_layer = Dense(latent_dim, activation=tf.keras.layers.LeakyReLU(), name='Dense_layer')(batch_layer1)
        batch_layer2 = BatchNormalization()(dense_layer)
        lstm_layer2 = LSTM(5, kernel_regularizer=kernel_regularizer, return_sequences=True,
                           activation=tf.keras.layers.LeakyReLU(), name='lstm_layer2')(batch_layer2)

        lstm_model = Model(lstm_input, lstm_layer2, name='lstm_model')

        lstm_model.compile(loss=loss_fn, optimizer=optimization,
                                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
        return lstm_model

    def train_model(self, batches, epochs):
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
    #save only 3 columns name, session number, and HSCL measurement
    HSCL_df = pd.DataFrame()
    HSCL_df['heb_code'] = sbs_df['heb_code']
    HSCL_df['session_n'] = sbs_df['session_n']
    HSCL_df['c_b_hscl'] = sbs_df['c_b_hscl']
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

    #convert HSCL to numpy for easier processing
    HSCL_arr = HSCL_df.to_numpy()

    #keep only patients with codes
    new_arr = np.empty((0,3))
    HSCL_ch_arr = np.empty((0,2))

    for i, row in enumerate(HSCL_arr):
        #get HSCL only from patients with congruence sessions
        if row[0] in list(filter_pat['heb_name']):
            ind_li = filter_pat.loc[(filter_pat['heb_name'] == row[0])]
            if str(row[1]) in list(ind_li['sess_n']):
                following = HSCL_arr[i+1]
                #if HSCL is NULL don't add the rows
                if row[2]=='#NULL!' or following[2]=='#NULL!':
                    continue
                new_arr = np.vstack([new_arr, row])
                new_arr = np.vstack([new_arr, following])
                #calculate change in hscl between the following and current sessions
                hscl_ch = following[2] - row[2]
                HSCL_ch_arr = np.vstack((HSCL_ch_arr, (row[0], hscl_ch)))

    #get unsorted unique values of patient names
    uniq, index = np.unique(HSCL_ch_arr[:, 0], return_index=True)
    uniq = uniq[index.argsort()]

    y_output = np.empty((0,5))
    for u in uniq:
        row_HSCL = HSCL_ch_arr[np.where(HSCL_ch_arr[:,0] == u), 1].astype('float')
        #if there are less than 5 calculated change in HSCL pre-post
        if len(row_HSCL.T)<5:
            row_HSCL = np.append(row_HSCL, np.median(row_HSCL))
        y_output = np.vstack((y_output, row_HSCL))
    y_output = np.hstack((uniq.reshape(len(uniq),1), y_output))

    #sort by patient names to make sure that indices match the indices in x_inp
    y_output = y_output[y_output[:, 0].argsort()]

    X_train, X_test, y_train, y_test = train_test_split(x_inp, y_output, test_size = 0.25, random_state = 42)

    class_lstm = lstm_cong(X_train[:,1:].astype('float32'), y_train[:,1:].astype('float32'))
    lstm_model = class_lstm.train_model(batches=10, epochs=512)

    x_test_float = X_test[:,1:].astype('float32')
    x_test_float = x_test_float.reshape(len(x_test_float), 1, len(x_test_float.T))
    y_test_float = y_test[:, 1:].astype('float32')
    y_test_float = y_test_float.reshape(len(x_test_float), 1, len(x_test_float.T))
    test_output = lstm_model.predict(x_test_float)
    evaluate_test = lstm_model.evaluate(x_test_float, y_test_float)
    model_weights = lstm_model.get_weights()


















