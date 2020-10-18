import pygame
import numpy as np
import random

pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Open Sans', 15)
screen = pygame.display.set_mode([500, 500])
values = np.zeros((5,5))
rects = []
perceptrons = []
training_inputs = []

class Perceptron(object):

  def __init__(self, no_of_inputs, learning_rate=0.01, iterations=100):
    self.iterations = iterations
    self.learning_rate = learning_rate
    self.no_of_inputs = no_of_inputs
    self.weights = np.random.rand(self.no_of_inputs + 1)
    self.weights = self.weights/10

  def train(self, training_data, labels):
    for _ in range(self.iterations):
      for input, label in zip(training_data, labels): # ZADANIE DOMOWE 2 - losowosc
        # input = noisy(input) # ZADANIE DOMOWE - zaburzenie wejscia
        prediction = self.output(input)
        self.weights[1:] += self.learning_rate * (label - prediction) * input
        self.weights[0] += self.learning_rate * (label - prediction)
        # ZADANIE DOMOWE 3 - warunek stopu
        # ZADANIE DOMOWE 4 - PLA + RPLA
        # h(x) = (ax + b) % m

  def trainSPLA(self, training_data, labels):
        allCorrect = False
        zippedList = list(zip(training_data, labels))

        while allCorrect == False:
            allCorrect = True
            for input, label in zippedList: # ZADANIE DOMOWE 2 - losowosc
                # input = noisy(input) # ZADANIE DOMOWE - zaburzenie wejscia
                prediction = self.output(input)
                err = label - prediction
                if err != 0:
                    self.weights[1:] += self.learning_rate * err * input
                    self.weights[0] += self.learning_rate * err
                    allCorrect = False

  def output(self, input):
    summation = np.dot(self.weights[1:], input) + self.weights[0]
    if summation > 0:
      activation = 1
    else:
      activation = 0
    return activation

def setPerceptrons():
    for _ in range(10):
        perceptrons.append(Perceptron(5*5))

    number = [[] for _ in range(10) ]
    number[0] = [
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,0.0,0.0,0.0,1.0],
        [1.0,0.0,0.0,0.0,1.0],
        [1.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ]
    number[1] = [
        [0.0,0.0,0.0,1.0,0.0],
        [0.0,0.0,1.0,1.0,0.0],
        [0.0,1.0,0.0,1.0,0.0],
        [1.0,0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,1.0,0.0]
    ]
    number[2] = [
        [0.0,1.0,1.0,1.0,0.0],
        [1.0,0.0,0.0,0.0,1.0],
        [0.0,0.0,0.0,1.0,0.0],
        [0.0,0.0,1.0,0.0,0.0],
        [1.0,1.0,1.0,1.0,1.0]
    ]
    number[3] = [
        [1.0,1.0,1.0,1.0,1.0],
        [0.0,0.0,0.0,0.0,1.0],
        [0.0,0.0,1.0,1.0,1.0],
        [0.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ]
    number[4] = [
        [1.0,0.0,0.0,0.0,0.0],
        [1.0,0.0,0.0,1.0,0.0],
        [1.0,1.0,1.0,1.0,1.0],
        [0.0,0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,1.0,0.0]
    ]
    number[5] = [
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,0.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,1.0,1.0],
        [0.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ]
    number[6] = [
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,0.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ]
    number[7] = [
        [1.0,1.0,1.0,1.0,1.0],
        [0.0,0.0,0.0,1.0,0.0],
        [0.0,0.0,1.0,0.0,0.0],
        [0.0,1.0,0.0,0.0,0.0],
        [1.0,0.0,0.0,0.0,0.0]
    ]
    number[8] = [
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ]
    number[9] = [
        [1.0,1.0,1.0,1.0,1.0],
        [1.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0],
        [0.0,0.0,0.0,0.0,1.0],
        [1.0,1.0,1.0,1.0,1.0]
    ]

    training_inputs = [ np.ravel(n) for n in number ]

    for i in range(10):
        labels = np.zeros(10)
        labels[i] = 1
        perceptrons[i].trainSPLA(training_inputs, labels)

def printPerceptronsOutput(): 
    for x in range(10):
        print(f'Perceptron {x}: {perceptrons[x].output(np.ravel(values))}')

def drawPerceptronsOutput():
    result = "Digits detected by perceptrons:"

    detected = False

    for x in range(10):
        if perceptrons[x].output(np.ravel(values)) == 1:
            result += f" {x},"
            detected = True

    if detected == False:
        result += " None."

    textsurface = myfont.render(result[:-1], False, (0, 0, 0)) #wypisanie bez ostatniego znaku - przecinka
    screen.blit(textsurface,(10,330))
    pygame.display.flip()

def draw_rectangles():
    left = 10
    top = 10
    width = 40
    height = 40

    rows, cols = values.shape
    for row in range(rows):
        for value in range(cols):
            rect = pygame.Rect(left,top,width,height)
            rects[row][value] = rect
            if values[row][value] == 0:
                pygame.draw.rect(screen, (0, 0, 0), rect)
            else:
                pygame.draw.rect(screen, (0, 0, 255), rect)
            left = left + 50
        left = 10
        top = top + 50

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
    pygame.display.flip()
    running = True
    
    setPerceptrons()
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                for row in range(len(rects)):
                    for col in range(len(rects[row])):
                        if rects[row][col].collidepoint(pos):
                            if values[row][col] == 0:
                                values[row][col] = 1
                            else:
                                values[row][col] = 0
                screen.fill((255, 255, 255))
                draw_rectangles()
                printPerceptronsOutput()
                drawPerceptronsOutput()

    pygame.quit()

if __name__ == "__main__":
    main()