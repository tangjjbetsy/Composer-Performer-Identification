import librosa
import pandas as pd
import numpy as np
import pickle

composer_selection = pd.read_csv("data/composer_selection.csv", header=0)
composer_selection['canonical_composer'] = composer_selection['canonical_composer'].astype("category")
composer_selection['composer_label'] = composer_selection['canonical_composer'].cat.codes.tolist()

data_length = len(composer_selection) 
label_length = len(composer_selection["canonical_composer"].value_counts())

X = np.zeros((int(16*data_length), 300, 64))
Y = np.zeros((int(16*data_length), label_length))

for i in range(data_length):
    performance = composer_selection.iloc[i]
    filepath = "data/"+performance['audio_filename']
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