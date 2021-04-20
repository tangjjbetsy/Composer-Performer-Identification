import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def norm(spec):
    mean = np.reshape(np.mean(spec, axis=1), (spec.shape[0],1))
    std = np.reshape(np.std(spec, axis=1), (spec.shape[0],1))
    spec = np.divide(np.subtract(spec,np.repeat(mean, spec.shape[1], axis=1)), np.repeat(std, spec.shape[1], axis=1))
    return spec

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

composer_selection = pd.read_csv("data/composer_selection.csv", header=0)
composer_selection['canonical_composer'] = composer_selection['canonical_composer'].astype("category")
composer_selection['composer_label'] = composer_selection['canonical_composer'].cat.codes.tolist()

data_length = len(composer_selection) 
label_length = len(composer_selection["canonical_composer"].value_counts())

X = np.zeros((int(16*data_length), 300, 64))
Y = np.zeros((int(16*data_length), label_length))

for i in tqdm(range(data_length)):
    performance = composer_selection.iloc[i]
    filepath = "/import/c4dm-datasets/maestro-v2.0.0/"+performance['audio_filename']
    label_id = performance['composer_label']
    wav, fs = librosa.load(filepath, sr=44100, offset=10, duration=60)
    mel = librosa.feature.melspectrogram(wav, 
                                         sr=fs, 
                                         hop_length=int(fs*0.0125),
                                         win_length=int(fs*0.025),
                                         n_fft=2048,
                                         n_mels=64, 
                                         fmax =8000)
    # mel = librosa.power_to_db(mel)
    mel = np.log(mel+1e-8)
    mel = norm(mel)

    mel = mel.transpose((1,0))
    mel = pad_trunc_seq(mel, 4800)
    mel = mel.reshape((16, 300, 64))
    
    y = np.zeros((1, label_length))
    y[0, label_id] = 1
    y = np.repeat(y, 16, axis=0)
    
    X[int(i*16):int((i+1)*16),] = mel
    Y[int(i*16):int((i+1)*16),] = y

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

np.save("X_train", X_train)
np.save("y_train", y_train)
np.save("X_test", X_test)
np.save("y_test", y_test)