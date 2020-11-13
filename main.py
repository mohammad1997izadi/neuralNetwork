

#===================================================================
#  This file is subject to the terms and conditions defined in     |
#  file 'LICENSE.txt', which is part of this source code package.  |
#===================================================================

from neural_network import *
import pandas as pd

dataset = pd.read_excel ('height_weight_dataset.xlsx')
gender = pd.DataFrame(dataset, columns= ['Gender'])
hW = pd.DataFrame(dataset, columns= ['Height', 'Weight'])
input = []


allTrueY = np.array(gender['Gender'])
for h, w in zip(hW['Height'], hW['Weight']):
  input.append([h, w])


network = NeuralNetwork(2)
network.training(input, allTrueY)
print('------------------------ female -----------------------------')
print(round(network.feedForward([64, 139])))
print('------------------------ male -----------------------------')
print(round(network.feedForward([72, 220])))
print('------------------------ femail -----------------------------')
print(round(network.feedForward([66, 150])))
print('------------------------ male -----------------------------')
print(round(network.feedForward([69, 207])))