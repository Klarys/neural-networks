import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Open Sans', 24)
screen = pygame.display.set_mode([550, 600])
values = np.zeros((10,10))
rects = []
perceptrons = []
number = [[] for _ in range(10) ]
number[0] = [
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[1] = [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[2] = [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[3] = [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[4] = [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[5] = [
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[6] = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[7] = [
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[8] = [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
number[9] = [
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]
training_inputs = []
training_inputs = [ np.ravel(n) for n in number ]
training_inputs_standarized = []
allLabels = []
no_of_ui_buttons = 16
buttonHeight = 20
startingWeights = []
buttonsPressed = []


def fourier_transform(x):
    a = np.abs(np.fft.fft(x))
    a[0] = 0
    return a/np.max(a)

def _standarise_features(x):
    return (x - np.mean(x))/np.std(x)

class Adaline(object):

    def __init__(self, no_of_input, learning_rate=0.01, iterations=100000, biased=False):
        self.no_of_input = no_of_input
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.random.random(2*self.no_of_input + 1)/10 # Zadanie domowe: dodanie biasu jest opcjonalne
        self.errors = []

    def _add_bias(self, x):
        if self.biased:
            return x.np.hstack()
        else:
            return x
    

    def train(self, training_data_x, training_data_y):
        zippedList = list(zip(training_data_x, training_data_y))
        for _ in range(self.iterations):
            e = 0

            randomIndex = random.randint(0, len(zippedList) - 1)

            x = zippedList[randomIndex][0]
            y = zippedList[randomIndex][1]
                # out = self.output(x)
                # print(x)
                # print("x after concat:")
                
                # print(x)
                # print(out)
                # print(y)
                # print(y-out)
            out = self.output(x)
            # print(self.learning_rate * (y - out) *x )
            self.weights[1:] += self.learning_rate * (y - out) * x * self.derivative(out) #Co gdy mamy funkcje aktywacji - zmiana pochodnej
            self.weights[0] += self.learning_rate * (y - out) * self.derivative(out)#* 1
            e += 0.5 * (y - out)**2
            self.errors.append(e)
        plt.plot(range(len(self.errors)), self.errors)
        plt.ylim(-0.5, 2.0)
        plt.savefig('errors.pdf')
        # plt.show()

    def activation(self, x): # Dodac funkcje aktywacji -> zmiana pochodnej
        return 1/(1 + np.exp(-x))#x -sigmoid (wymaga zmiany pochodnej czastkowej - patrz wyzej)

    def derivative(self, x):
        return 1/(1 + np.exp(-x)) * (1 - (1/(1 + np.exp(-x))))
        
  
    def output(self, input):
            # inp = self._standarise_features(input)
            # inp = np.concatenate([inp, fourier_transform(inp)])
            summation = self.activation(np.dot(self.weights[1:], input) + self.weights[0])
            # print("dot + activation:")
            # print(summation)
            return summation

def setPerceptrons():
    global training_inputs
    global training_inputs_standarized

    print("BE:")
    print(training_inputs)

    for input in training_inputs:
        training_inputs_standarized.append(np.concatenate([input, fourier_transform(input)]))
        # training_inputs_standarized.append(np.concatenate([_standarise_features(input), fourier_transform(_standarise_features(input))]))
        # training_inputs_standarized.append(_standarise_features(np.concatenate([input, fourier_transform(input)])))

    print("AF:")
    print(training_inputs_standarized)

    for _ in range(10):
        perceptrons.append(Adaline(10*10))

    for i in range(10):
        labels = np.zeros(10)
        labels[i] = 1
        allLabels.append(labels)
        perceptrons[i].train(training_inputs_standarized, labels)

def printPerceptronsOutput(): 
    for x in range(10):
        # print(f'Perceptron {x}: {perceptrons[x].output(np.concatenate([_standarise_features(np.ravel(values)), fourier_transform(_standarise_features(np.ravel(values)))]))}')
        # print(f'Perceptron {x}: {perceptrons[x].output(_standarise_features(np.concatenate([np.ravel(values), fourier_transform(np.ravel(values))])))}')
        print(f'Perceptron {x}: {perceptrons[x].output(np.concatenate([np.ravel(values), fourier_transform(np.ravel(values))]))}')

def drawPerceptronsOutput():
    result = ""
    detected = False

    max = -1000
    maxIndex = -1
    for x in range(10):
        # _output = perceptrons[x].output((np.concatenate([_standarise_features(np.ravel(values)), fourier_transform(_standarise_features(np.ravel(values)))])))
        # _output = perceptrons[x].output(_standarise_features(np.concatenate([np.ravel(values), fourier_transform(np.ravel(values))])))
        _output = perceptrons[x].output((np.concatenate([np.ravel(values), fourier_transform(np.ravel(values))])))
        if _output > 0.00: # prog
            if _output > max:
                max = _output
            # result += f" {x},"
                maxIndex = x
            detected = True

    if detected == False:
        result += " None."
    else:
        result += f"{maxIndex}"

    textSurface1 = myfont.render("Digits detected by perceptrons:", False, (0, 0, 0))
    textSurface2 = myfont.render(result, False, (0, 0, 0)) #wypisanie bez ostatniego znaku - przecinka
    screen.blit(textSurface1,(10,330))
    screen.blit(textSurface2,(10,350))
    pygame.display.flip()

def drawMessage(message):
    textSurface = myfont.render(message, False, (0, 0, 0))
    screen.blit(textSurface,(10,500))
    pygame.display.flip()

def drawUI():
    buttonTexts = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Reset", "<-", "->", "^", "v", "Generate noise"]
    
    for buttonNumber in range(len(buttonTexts)):
        pygame.draw.rect(screen, (102, 153, 255), pygame.Rect(300,10 + (buttonHeight+15)*buttonNumber,170,buttonHeight))
        textsurface = myfont.render(buttonTexts[buttonNumber], False, (0, 0, 0))
        screen.blit(textsurface,(310,10 + (buttonHeight+15)*buttonNumber))

    for buttonNumber in range(10):
        pygame.draw.rect(screen, (2, 1, 116), pygame.Rect(480,10 + (buttonHeight+15)*buttonNumber,30,buttonHeight))
        textsurface = myfont.render(" + ", False, (230, 230, 230))
        screen.blit(textsurface,(487,10 + (buttonHeight+15)*buttonNumber))

    pygame.display.flip()

def detectButtonClicked(x, y): #returns a number of button clicked
    startHeight = 10
    for buttonNumber in range(no_of_ui_buttons):
        buttonHeightStart = startHeight + (buttonHeight+15)*buttonNumber
        buttonHeightEnd = startHeight + (buttonHeight+15)*buttonNumber + buttonHeight
        if(y>= buttonHeightStart and y<=buttonHeightEnd):
            if(x>=300 and x<=470):
                print(f'Button number {buttonNumber} clicked')
                activateButton(buttonNumber)
                return buttonNumber
            elif(buttonNumber <10 and x>=487 and x<=517):
                print(f'Button "+" number {buttonNumber} clicked')
                addExample(buttonNumber)
    return -1

def addExample(buttonNumber):
    global allLabels
    labels = np.zeros(10)
    labels[buttonNumber] = 1

    global values
    global training_inputs
    training_inputs.append(np.ravel(values))

    for i in range(10):
        perceptrons[i].weights = np.random.rand(perceptrons[i].no_of_inputs + 1)
        perceptrons[i].weights = perceptrons[i].weights/10
    
    for i in range(10):
        labelsCopy = allLabels[i].copy()
        print("Range:")
        print(range(len(buttonsPressed)))
        for j in range(len(buttonsPressed)):
            if i == buttonsPressed[j]:
                newLabel = 1.0
            else:
                newLabel = 0.0
            labelsCopy = np.append(labelsCopy, newLabel)
        perceptrons[i].train(training_inputs, labelsCopy)

def buttonZero():
    print("button Zero clicked")
    global values
    buttonsPressed.append(0)
    values = np.array([
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonOne():
    print("button One clicked")
    global values
    buttonsPressed.append(1)
    values = np.array([
[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
])

def buttonTwo():
    print("button Two clicked")
    global values
    buttonsPressed.append(2)
    values = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonThree():
    print("button Three clicked")
    global values
    buttonsPressed.append(3)
    values = np.array([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonFour():
    print("button Four clicked")
    global values
    buttonsPressed.append(4)
    values = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonFive():
    print("button Five clicked")
    global values
    buttonsPressed.append(5)
    values = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonSix():
    print("button Six clicked")
    global values
    buttonsPressed.append(6)
    values = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonSeven():
    print("button Seven clicked")
    global values
    buttonsPressed.append(7)
    values = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonEight():
    print("button Eight clicked")
    global values
    buttonsPressed.append(8)
    values = np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonNine():
    print("button Nine clicked")
    global values
    buttonsPressed.append(9)
    values = np.array([
        [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

def buttonReset():
    print("button Reset clicked")
    for i in range(10):
        perceptrons[i].weights = np.random.random(2*perceptrons[i].no_of_input + 1)/10
    global values
    values = np.zeros((10,10))
    global buttonsPressed
    buttonsPressed = []

    for i in range(10):
        labels = np.zeros(10)
        labels[i] = 1
        perceptrons[i].train(training_inputs_standarized, labels)

def buttonLeft():
    print("button Left clicked")
    rows, cols = values.shape
    print(cols)
    for row in range(rows):
        for col in range(1, cols):
            print(f'Row: {row}, col: {col}')
            if values[row][col] == 1:
                values[row][col-1] = 1.0
                values[row][col] == 0.0
            else:
                values[row][col-1] = 0.0
        values[row][cols-1] = 0.0

def buttonRight():
    print("button Right clicked")
    rows, cols = values.shape
    print(cols)
    for row in range(rows):
        for col in reversed(range(cols-1)):
            print(f'Row: {row}, col: {col}')
            if values[row][col] == 1:
                values[row][col+1] = 1.0
                values[row][col] == 0.0
            else:
                values[row][col+1] = 0.0
        values[row][0] = 0.0

def buttonUp():
    print("button Up clicked")
    rows, cols = values.shape
    print(cols)
    for row in range(1, rows):
        for col in range(cols):
            print(f'Row: {row}, col: {col}')
            if values[row][col] == 1:
                values[row-1][col] = 1.0
                values[row][col] == 0.0
            else:
                values[row-1][col] = 0.0

    for col in range(cols):
        values[rows-1][col] = 0.0

def buttonDown():
    print("button Down clicked")
    rows, cols = values.shape
    print(cols)
    for row in reversed(range(0, rows-1)):
        for col in range(cols):
            print(f'Row: {row}, col: {col}')
            if values[row][col] == 1:
                values[row+1][col] = 1.0
                values[row][col] == 0.0
            else:
                values[row+1][col] = 0.0

    for col in range(cols):
        values[0][col] = 0.0

def buttonNoise():
    print("button Noise clicked")
    rows, cols = values.shape
    for row in range(rows):
        for value in range(cols):
            if random.random() <= 0.02:
                values[row][value] = (-1) * values[row][value] + 1
                    

def activateButton(buttonNumber):
    switcher = {
        0: buttonZero,
        1: buttonOne,
        2: buttonTwo,
        3: buttonThree,
        4: buttonFour,
        5: buttonFive,
        6: buttonSix,
        7: buttonSeven,
        8: buttonEight,
        9: buttonNine,
        10: buttonReset,
        11: buttonLeft,
        12: buttonRight,
        13: buttonUp,
        14: buttonDown,
        15: buttonNoise
    }
    # Get the function from switcher dictionary
    func = switcher.get(buttonNumber, lambda: "Invalid button number")
    # Execute the function
    func()

def draw_rectangles():
    left = 10
    top = 10
    width = 20
    height = 20

    rows, cols = values.shape
    for row in range(rows):
        for value in range(cols):
            rect = pygame.Rect(left,top,width,height)
            rects[row][value] = rect
            if values[row][value] == 0:
                pygame.draw.rect(screen, (0, 0, 0), rect)
            else:
                pygame.draw.rect(screen, (0, 0, 255), rect)
            left = left + 25
        left = 10
        top = top + 25

    pygame.display.flip()

def init_rectangles():
    left = 10
    top = 10
    width = 40
    height = 40

    rows, cols = values.shape
    for row in range(rows):
        rects_tmp = []
        for value in range(cols):
            rect = pygame.Rect(left,top,width,height)
            rects_tmp.append(rect)
            left = left + 50
        left = 10
        top = top + 50
        rects.append(rects_tmp)

def main():
    
    screen.fill((255, 255, 255))
    init_rectangles()
    draw_rectangles()
    drawUI()
    pygame.display.flip()
    running = True
    
    setPerceptrons()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                print(values)
                detectedButton = detectButtonClicked(pos[0], pos[1])
                screen.fill((255, 255, 255))
                for row in range(len(rects)):
                    for col in range(len(rects[row])):
                        if rects[row][col].collidepoint(pos):
                            if values[row][col] == 0:
                                values[row][col] = 1
                            else:
                                values[row][col] = 0
                printPerceptronsOutput()
                drawPerceptronsOutput()
                if detectedButton == 10:
                    drawMessage("Reset finished.")
                draw_rectangles()
                drawUI()
                
    pygame.quit()

if __name__ == "__main__":
    main()