import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length. 
    
    Args:
      x: ndarray, input sequence data. 
      max_len: integer, length of sequence to be padded or truncated. 
      
    Returns:
      ndarray, Padded or truncated input sequence data. 
    """
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]

    return x_new

def process_data(X, Y, label_length, indices, data):
    n = 0
    for i in tqdm(indices):
        performance = data.iloc[i]
        filepath = "/import/c4dm-datasets/maestro-v2.0.0/"+performance['audio_filename']
        label_id = performance['performer_id']
        wav, fs = librosa.load(filepath, sr=44100, offset=0, duration=500)
        mel = librosa.feature.melspectrogram(wav, 
                                            sr=fs, 
                                            hop_length=1024,
                                            win_length=2048,
                                            n_fft=2048,
                                            n_mels=64, 
                                            fmax =8000)
        # cqt = librosa.cqt(wav, 
        #                   sr=fs, 
        #                   hop_length=int(fs*0.0125),
        #                   win_length=int(fs*0.025),
        #                   n_fft=2048,
        #                   n_bins=80)
        mel = librosa.power_to_db(mel)
        # mel = np.log(mel+1e-8)
        mel = mel.transpose((1,0))
        mel = pad_trunc_seq(mel, 20736) #(20376, 64)
        mel = mel.reshape((1,1152,1152))
        
        y = np.zeros((1, label_length))
        y[0, label_id] = 1
        # y = np.repeat(y, 10, axis=0)
        
        # X[int(n*10):int((n+1)*10)] = mel
        # Y[int(n*10):int((n+1)*10)] = y
        X[n] = mel
        Y[n] = y

        n += 1

    return X, Y

def normalization(X):
    scaler = StandardScaler()
    scaler.fit(X.reshape((-1,64)))
    X = scaler.transform(X.reshape((-1,64)))
    X = X.reshape((-1,1,2150, 64))
    return X

# composer_selection = pd.read_csv("data/composer_selection.csv", header=0)
# composer_selection['canonical_composer'] = composer_selection['canonical_composer'].astype("category")
# composer_selection['composer_label'] = composer_selection['canonical_composer'].cat.codes.tolist()

performer_selection = pd.read_csv("data/performer_selection.csv", header=0)
performer_selection['performer'] = performer_selection['performer'].astype("category")
performer_selection['performer_id'] = performer_selection['performer'].cat.codes.tolist()

data_length = len(performer_selection) 
print(data_length)
label_length = len(performer_selection["performer_id"].value_counts())

# #indexes matrix
# indices = np.arange(300).reshape(5,60)
# train_indices = indices[np.ix_(np.arange(5), np.arange(50))].flatten()
# test_indices = indices[np.ix_(np.arange(5), np.arange(50,55))].flatten()
# val_indices = indices[np.ix_(np.arange(5), np.arange(55,60))].flatten()

# train_length = int(data_length*10*0.8)
# test_length = int(data_length*10*0.1)
# val_length = int(data_length*10*0.1)

# train_length = 250
# test_length = 25
# val_length = 25

# X_train = np.zeros((train_length, 1, 1152, 1152))
# Y_train = np.zeros((train_length, label_length))
# X_val = np.zeros((val_length, 1, 1152, 1152))
# Y_val = np.zeros((val_length, label_length))
# X_test = np.zeros((test_length, 1, 1152, 1152))
# Y_test = np.zeros((test_length, label_length))

# X = np.zeros((int(data_length*10), 1, 2150, 64))
# Y = np.zeros((int(data_length*10), label_length))

X = np.zeros((data_length, 1, 1152, 1152))
Y = np.zeros((data_length, label_length))
print(data_length)
X, Y = process_data(X, Y, label_length, np.arange(data_length), performer_selection)
# X = normalization(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=40, test_size=0.2)

X_test, X_val, Y_test, Y_val = train_test_split(X_test,Y_test, random_state=42, test_size=0.5)

# X_train, Y_train = process_data(X_train, Y_train, label_length, train_indices, composer_selection)
# X_train = normalization(X_train)
np.save("X_train_p", X_train)
np.save("y_train_p", Y_train)

# np.save("X_p", X)
# np.save("Y_p", Y)

# X_val, Y_val = process_data(X_val, Y_val, label_length, val_indices, composer_selection)
# X_val = normalization(X_val)
np.save("X_val_p", X_val)
np.save("y_val_p", Y_val)

# X_test, Y_test = process_data(X_test, Y_test, label_length, test_indices, composer_selection)
# X_test = normalization(X_tes/t)
np.save("X_test_p", X_test)
np.save("y_test_p", Y_test)




