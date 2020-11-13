
#===================================================================
#  This file is subject to the terms and conditions defined in     |
#  file 'LICENSE.txt', which is part of this source code package.  |
#===================================================================

import numpy as np
from functions import * 


class NeuralNetwork:
    
    def __init__(self, inputNumb):
        self.inputNumb = inputNumb
        self.w1 = []  
        self.w2 = []  
        self.w3 = []  

        self.wo = []  

        for counter in range(self.inputNumb):
            self.w1.append(np.random.normal())
            self.w2.append(np.random.normal())
            self.w3.append(np.random.normal())
          

        for counter in range(3):
            self.wo.append(np.random.normal())

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
       
        self.bo = np.random.normal()
        
    def feedForward(self, x):

        n1 = 0
        n2 = 0
        n3 = 0
      

        for counter in range(self.inputNumb):
            n1 += self.w1[counter] * x[counter]
            n2 += self.w2[counter] * x[counter]
            n3 += self.w3[counter] * x[counter]
           
        n1 = sigmoid(n1 + self.b1)
        n2 = sigmoid(n2 + self.b2)
        n3 = sigmoid(n3 + self.b3)
        
        no = sigmoid(self.wo[0] * n1 + self.wo[1] * n2 + self.wo[2] * n3 + self.bo)
        return no

    def training(self, input, allTrueY):
        learnRate = 0.001
        iterationNumber = 1000

        n1 = 0
        n2 = 0
        n3 = 0  
        for iteration in range(iterationNumber):

            for x, y_true in zip(input, allTrueY):

                
                for counter in range(self.inputNumb):
                    n1 += self.w1[counter] * x[counter]
                    n2 += self.w2[counter] * x[counter]
                    n3 += self.w3[counter] * x[counter]
                

                    
                sum_n1 = n1 + self.b1
                sum_n2 = n2 + self.b2
                sum_n3 = n3 + self.b3
                

                n1 = sigmoid(sum_n1)
                n2 = sigmoid(sum_n2)
                n3 = sigmoid(sum_n3)

                sum_no = self.wo[0] * n1 + self.wo[1] * n2 + self.wo[2] * n3 + self.bo
                no = sigmoid(sum_no)

                y_pred = no

                dL_dYPred = -2 * (y_true - y_pred)

                dYPred_dWO = []

                dYPred_dWO.append(n1 * derivSigmoid(sum_no))
                dYPred_dWO.append(n2 * derivSigmoid(sum_no))
                dYPred_dWO.append(n3 * derivSigmoid(sum_no))


                d_ypred_d_bo = derivSigmoid(sum_no)

                dYPred_dN1 = self.wo[0] * derivSigmoid(sum_no)
                dYPred_dN2 = self.wo[1] * derivSigmoid(sum_no)
                dYPred_dN3 = self.wo[2] * derivSigmoid(sum_no)

                dN1_dW = []
                dN2_dW = []
                dN3_dW = []
             
                for counter in range(self.inputNumb):
                    dN1_dW.append(x[counter] * derivSigmoid(sum_n1))
                    dN2_dW.append(x[counter] * derivSigmoid(sum_n2))
                    dN3_dW.append(x[counter] * derivSigmoid(sum_n3))
                    

                dN1_dB1 = derivSigmoid(sum_n1)
                dN2_dB2 = derivSigmoid(sum_n2)
                dN3_dB3 = derivSigmoid(sum_n3)


                for counter in range(self.inputNumb):
                    self.w1[counter] -= (learnRate * dL_dYPred * dYPred_dN1 * dN1_dW[counter])
                    self.w2[counter] -= (learnRate * dL_dYPred * dYPred_dN2 * dN2_dW[counter])
                    self.w3[counter] -= (learnRate * dL_dYPred * dYPred_dN3 * dN3_dW[counter])
                    

                for counter in range(3):
                    self.wo[counter] -= (learnRate * dL_dYPred * dYPred_dWO[counter])
                    
                self.b1 -=  (learnRate * dL_dYPred * dYPred_dN1 * dN1_dB1)

                self.b2 -= (learnRate * dL_dYPred * dYPred_dN2 * dN2_dB2)
                self.b3 -= (learnRate * dL_dYPred * dYPred_dN3 * dN3_dB3)
               
                self.bo -= (learnRate * dL_dYPred * d_ypred_d_bo)

            if iteration % 10 == 0:
                print(self.feedForward([73, 241]))   
                print(self.feedForward([64, 131]))  
                print('==============================')  
