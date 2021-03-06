from keras.models import model_from_json
from spectrogram_generator import Loop
import requests
import numpy as np

self = Loop()
file = open('model.json')
model = model_from_json(file.read())
file.close()

model.load_weights('model_15_27.h5')

# Testing a new image
for k in range(1):
    x = self.TestData()
    test_image = np.zeros((1,1,129,178))
    test_image[0,0,:,:] = x


    # Predicting the test image
    score = model.predict(test_image)
    #print(model.predict_classes(test_image))

    threatProbability = np.diff(score)[0][0]
    # if threatProbability > 0.7:
    if score[0][0] > 0.7:
        message = 'Threat Detected'
        status = 'unsafe'
    else:
        message = 'safe'
        status = 'safe'
    print '  '
    print 'threatMeasure = ', threatProbability, message, score[0][0]

    #requests.post('http://192.168.1.106:8000/ping/', data={'atmId': 'A17HYD1875', 'threat': str(message), 'status': status})


