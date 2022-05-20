# Import libraries
import os, sys
import numpy as np
import pandas as pd
import librosa
import IPython.display as ipd
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display

# Get the base directory
basedir = os.getcwd()
dirname = basedir+ "/Data/genres_original"

# Save audio paths and labels
audio_paths = []
# audio_dict = {}
audio_label = []
# Print all the files in different directories
for root, dirs, files in os.walk(dirname, topdown=False):
    for filenames in files:
        if filenames.find('.wav') != -1:

            audio_paths.append(os.path.join(root, filenames))
            filenames = filenames.split('.', 1)
            filenames = filenames[0]
            audio_label.append(filenames)
audio_paths = np.array(audio_paths)
audio_label = np.array(audio_label)
audio_paths.shape


#########################################################################
## The following code is for visualizing features using a single audio ##
#########################################################################
# audio_1 = audio_paths[0]
# print(audio_1)
# audio = '/Users/namratadutt/Downloads/03-01-01-01-01-01-01.wav'
# y, sr = librosa.load(audio_1)
# print(y.shape)

## Works in jupyter notebook
# ipd.Audio(audio_1)

# fig, ax = plt.subplots(1, figsize= (15, 5))
# librosa.display.waveshow(y, sr=sr)


#display Spectrogram
'''
X = librosa.stft(y)
print(X.shape)
# Amplitude to decibels
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
# converting to log(Power Spectrogram)
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log', cmap = 'plasma') 
plt.colorbar()
'''

# Mel-Spectrogram using pre-computed Spectrogram
'''
X = np.abs(X)**2
S = librosa.feature.melspectrogram(S=X, sr=sr)
S_db = librosa.power_to_db(S)
print(S_db.shape)
plt.figure(figsize=(14, 5))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap = 'plasma') 
plt.colorbar()
'''

# Mel-Spectrogram using raw signal
'''
M = librosa.feature.melspectrogram(y=y)
M_db = librosa.power_to_db(M)
plt.figure(figsize=(14, 5))
librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel', cmap = 'plasma')
plt.colorbar()
'''

# Check the difference
# np.allclose(S_db, M_db)

# MFCC (Mel-frequency cepstral coefficients)
# Compressed features of a Mel-spectrogram
# The higher order coefficients represent the excitation information, or the periodicity in the waveform, 
# while the lower order cepstral coefficients represent the vocal tract shape or smooth spectral shape
'''
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 10)
print(mfcc.shape)
# mfcc_db = librosa.power_to_db(mfcc, ref=np.max)
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfcc, sr=sr, x_axis='time')
plt.colorbar()
'''

# Zero-crossings
'''
plt.figure(figsize=(14, 5))
# librosa.display.waveshow(y, sr=sr)
plt.plot(y)
plt.grid()
zc = librosa.zero_crossings(y, pad = False)
print(zc.shape)
print(sum(zc))
'''

'''
# Zero-crossing rate
plt.figure(figsize=(14,5))
zcr = librosa.feature.zero_crossing_rate(y)
print(zcr.shape)
# Plot the zero-crossing rate
plt.plot(zcr[0])
'''

# Spectral centroid using raw signal
# sp_cen = librosa.feature.spectral_centroid(y=y, sr=sr)
# print(sp_cen.shape)


'''
X = librosa.stft(y)
times = librosa.times_like(sp_cen)
fig, ax = plt.subplots(1, figsize= (15, 5))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X), ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
ax.plot(times, sp_cen.T, label='Spectral centroid', color='w')
ax.legend(loc='upper right')
ax.set(title='log Power spectrogram')
'''

'''
S, phase = librosa.magphase(librosa.stft(y=y))
sp_cen1 = librosa.feature.spectral_centroid(S=S)
print(sp_cen1)
times = librosa.times_like(sp_cen1)
fig, ax = plt.subplots(1, figsize= (15, 5))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax)
ax.plot(times, sp_cen1.T, label='Spectral centroid', color='w')
ax.legend(loc='upper right')
ax.set(title='log Power spectrogram')
'''

# np.allclose(sp_cen, sp_cen1)

#Chromagrams
# chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
# print(chroma_stft.shape)


# fig, ax = plt.subplots(1, figsize= (15, 5))
# librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax)

#########################################################################
#########################################################################

# Create empty arrays to save the features
AllSpec = np.empty([1000, 1025, 1293])
AllMel = np.empty([1000, 128, 1293])
AllMfcc = np.empty([1000, 10, 1293])
AllZcr = np.empty([1000, 1293])
AllCen = np.empty([1000, 1293])
AllChroma = np.empty([1000, 12, 1293])

count = 0
bad_index = []
for i in tqdm(range(len(audio_paths))):
    try:

        path = audio_paths[i]
        y, sr = librosa.load(path)
        # For Spectrogram
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        AllSpec[i] = Xdb
        
        # Mel-Spectrogram 
        M = librosa.feature.melspectrogram(y=y)
        M_db = librosa.power_to_db(M)
        AllMel[i] = M_db
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc= 10)
        AllMfcc[i] = mfcc
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        AllZcr[i] = zcr
        
        # Spectral centroid
        sp_cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        AllCen[i] = sp_cen
        
        # Chromagram
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        AllChroma[i] = chroma_stft

        

    except Exception as e:
        bad_index.append(i)

# Delete audio at corrupt indices
AllSpec = np.delete(AllSpec, bad_index, 0)
AllMel = np.delete(AllMel, bad_index, 0)
AllMfcc = np.delete(AllMfcc, bad_index, 0)
AllZcr = np.delete(AllZcr, bad_index, 0)
AllCen = np.delete(AllCen, bad_index, 0)
AllChroma = np.delete(AllChroma, bad_index, 0)

# Convert to float32
AllSpec = AllSpec.astype(np.float32)
AllMel = AllMel.astype(np.float32)
AllMfcc = AllMfcc.astype(np.float32)
AllZcr = AllZcr.astype(np.float32)
AllCen = AllCen.astype(np.float32)
AllChroma = AllChroma.astype(np.float32)

# Delete labels at corrupt indices
audio_label = np.delete(audio_label, bad_index)

# Convert labels from string to numerical
audio_label[audio_label == 'blues'] = 0
audio_label[audio_label == 'classical'] = 1
audio_label[audio_label == 'country'] = 2
audio_label[audio_label == 'disco'] = 3
audio_label[audio_label == 'hiphop'] = 4
audio_label[audio_label == 'jazz'] = 5
audio_label[audio_label == 'metal'] = 6
audio_label[audio_label == 'pop'] = 7
audio_label[audio_label == 'reggae'] = 8
audio_label[audio_label == 'rock'] = 9
audio_label = [int(i) for i in audio_label]
audio_label = np.array(audio_label)

# Convert labels from numerical to categorical data
y = tensorflow.keras.utils.to_categorical(audio_label,num_classes = 10, dtype ="int32")

#print(AllSpec.shape)
#print(AllMel.shape)
#print(AllMfcc.shape)
#print(AllZcr.shape)
#print(AllCen.shape)
#print(AllChroma.shape)
#print(y.shape)

# Save all the features and labels in a .npz file
np.savez_compressed(os.getcwd()+"/MusicFeatures.npz", spec= AllSpec, mel= AllMel, mfcc= AllMfcc, zcr= AllZcr, cen= AllCen, chroma= AllChroma, target=y)
