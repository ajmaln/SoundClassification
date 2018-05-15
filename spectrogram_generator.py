import librosa
import numpy as np
# import matplotlib
# matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
plt.ion()
plt.style.use('ggplot')

import threading
import time

#
# def loop1_10():
#     for i in range(1, 11):
#         time.sleep(1)
#         print(i)
#
# threading.Thread(target=loop1_10).start()

# import alsaaudio, time, audioop

import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal
import os
# plt.plot(1)


class Loop:

    def __init__(self):
        self.frameSize = 8000
        self.runCommand = 0
        self.timeWindow = 5
        self.sxxDen = 1.0
        # plt.plot(1)

    def InitAudioStream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=8000, input=True, frames_per_buffer=self.frameSize)

    def CloseAudioStream(self):
        self.stream.stop_stream()
        time.sleep(0.1)
        self.stream.close()
        time.sleep(0.1)
        self.p.terminate()

    def ReadATestCase(self):
        self.InitAudioStream()
        # do this as long as you want fresh samples
        self._stringData = self.stream.read(self.frameSize * self.timeWindow)
        self.frame = np.frombuffer(self._stringData, dtype=np.int16)
        self.frame = self.frame * 1.0 / np.sqrt(np.mean(np.square(self.frame * 1.0)))
        self.CloseAudioStream()
        return self.frame

    def St(self):
        if self.runCommand == 1:
            print 'Thread already running, stoping and starting'
            self.Sp()
            time.sleep(0.1)

        self.InitAudioStream()
        self.runCommand = 1
        self.loopThread = threading.Thread(target=self.FrameInLoop).start()
        print 'Looping started'

    def Spectrogram(self):
        self.f, self.t, self.sxx = signal.spectrogram(self.frame.flatten(), fs = 8000)
        self.sxx = self.sxx / self.sxxDen
        #plt.clf()
        #time.sleep(0.1)
        #self.pltSpect = plt.imshow(self.sxx)
        #plt.pause(0.1)

    def TestData(self):
        self.ReadATestCase()
        self.Spectrogram()
        return self.sxx

    def MakeSafeDataSet(self, category, yClass):
        self._cwd = os.getcwd()
        self._pathSound = self._cwd + '/sound_data/' + category + '/'
        self._files = os.listdir(self._pathSound)
        # self.sound = []
        self.sound = np.array([])
        for self._file in self._files:
            self._sound, self.fs = librosa.load(self._pathSound + self._file, sr=8000)
            self._sound = self._sound / np.sqrt(np.mean(np.square(self._sound)))
            # self.sound = self.sound + list(self._sound)
            # print(self._sound, self.sound)
            self.sound = np.append(self.sound, self._sound)

        self.f, self.t, self.sxx = signal.spectrogram(self.sound, fs=8000)
        self.sxx = self.sxx #/ self.sxx.max()
        self.frameCount = int(round(np.floor((self.sxx.shape[1] / 178.0))))
        self.trainX = np.zeros((self.frameCount, 129, 178))
        self.trainY = np.zeros((self.frameCount, 2))
        for k in range(self.frameCount):
            self._k = k
            self.trainX[k, :, :] = self.sxx[:, k * 178: 178 * (k + 1)]
            self.trainX[k, :, :] = self.trainX[k, :, :] / self.sxxDen
            # self.trainX[k, :, :, 1] = librosa.feature.delta(self.trainX[k,:,:,0])
            self.trainY[k, :] = yClass
        return self.trainX, self.trainY

    def TrainingDataFromDir(self):
        xt, yt = self.MakeSafeDataSet('threat', [1, 0])
        print('Threat case read')
        xs, ys = self.MakeSafeDataSet('safe' , [0, 1])
        print('Safe case read')
        xTrain = np.concatenate((xt, xs), axis=0)
        yTrain = np.concatenate((yt, ys), axis=0)
        return xTrain, yTrain

    def Sp(self):
        self.runCommand = 0
        # time.sleep(0.1)
        self.CloseAudioStream()
        print 'Looping Stoped'

    def Plot(self):
        plt.clf()
        plt.plot(self.frame)

    def Predict(self):
        self.Spectrogram()
        self.xForPredict = np.zeros((1,) + self.sxx.shape + (1,))
        self.xForPredict[0,:,:,0] = self.sxx
        return self.xForPredict


if __name__ == '__main__':
    self = Loop()
