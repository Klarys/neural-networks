import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from examples import Point
from examples import Examples
from nn import Neural_Network

pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Open Sans', 24)
screen = pygame.display.set_mode([500, 500])

def main():
    
    ex = Examples()
    e = ex.generate(1000)
    print(e)
    np.max(e[0]), np.max(e[1])
    x_train = (np.array(e[0]) + 20)/40.0*0.8 + 0.1
    y_train = np.array(e[1])/3.14*0.8 + 0.1
    NN = Neural_Network()
    for i in range(1000):
        NN.train(x, y)
    
    err = NN.errors
    plt.plot(range(len(err)), err)
    plt.show()

    screen.fill((255, 255, 255))
    image = pygame.image.load('robot.png')
    screen.blit(image, (0, 0))
    pygame.display.flip()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pos[0] > 250:
                    print(pos)
                
    pygame.quit()

if __name__ == "__main__":
    main()