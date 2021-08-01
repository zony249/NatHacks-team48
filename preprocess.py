import os

import numpy as np
import pandas as pd
import mne
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import pylab
from PIL import Image

LABELS_PATH = os.path.join("rest_label", "participants.tsv")
DATA_PATH = "rest_dataset"
PREPROCESSED = "rest_dataset_preproc"


class Data_Index():
    def __init__(self):
        self.trainset ={}
        self.valset = {}
        self.labels = {}

    def populate(self, data_path=DATA_PATH, labels_path=LABELS_PATH, preproc_path=PREPROCESSED, preprocess_all=False):
        
        for subject_path in glob.iglob(data_path + "/" + "sub*"):
            #print(subject_path)
            subj_name = subject_path.split("/")[-1]
            set_path = glob.glob(subject_path + "/eeg/*.set")

            self.trainset[subj_name] = []
            for i, pth in enumerate(set_path): # pth is the original path of the .set file in the data_path
                preprocessed_path = os.path.join(preproc_path, subj_name + "_" +  str(i)) # new path
                self.trainset[subj_name].append(preprocessed_path)
                if preprocess_all == True:
                    self.preprocess_step(pth, preprocessed_path)
                
    def preprocess_step(self, old_path, new_path):
        raw = mne.io.read_raw_eeglab(old_path)
        sf = raw.info["sfreq"]
        data, times = raw[:, int(sf*0):]
        data = data.T
        t_seconds = times.shape[-1]/sf
        for i in range(int(t_seconds)):
            start = int(sf * i)
            end = int(sf * (i + 8))

            snippet = np.zeros((end-start, 64))
            
            if end > times.shape[-1]:

                snippet[:times.shape[-1]-start, :] = data[start:, :64]
                remaining = end - times.shape[-1]
                snippet[times.shape[-1]-start:, :] = data[:remaining, :64]
                end = remaining
                print(times.shape[-1]-start, remaining)
            else:
                snippet[:, :] = data[start:end, :64]
            ###################
            # USE FFT
            ###################
            
            output = np.zeros((128, 128, 64))
            for j in range(64):
                output[:, :, j] = spectrogram(snippet[:, j], sf)
                
            savepath = new_path + "_" + str(i) + ".npy"
            np.save(savepath, output)


def spectrogram(c, sf):

    assert len(c.shape) == 1
    N = int(sf*8) # number of samples
    f, t, Sxx = signal.spectrogram(c, fs=sf, nperseg = 175, noverlap=125, nfft=None, scaling="spectrum", mode="magnitude")
            #print(Sxx.shape)                
            #Sxx, f, t, image = plt.specgram(snippet[:,1], Fs=sf, )
            #plt.pcolormesh(t, f, Sxx)
            #Sxx = Sxx/np.amax(Sxx)
            #print(np.amax(Sxx))
    Sxx = np.uint8(Sxx / np.amax(Sxx) * 255)
    Sxx = Image.fromarray(Sxx).resize((128, 128)).convert("RGB")
    return np.array(Sxx)[:, :, 0]


if __name__ == "__main__":
    data_index = Data_Index()
    data_index.populate(preprocess_all=False)
