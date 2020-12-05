import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Open Sans', 24)
screen = pygame.display.set_mode([500, 500])

def main():
    
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
                if pos[0] >= 250:
                    print(pos)
                
    pygame.quit()

if __name__ == "__main__":
    main()