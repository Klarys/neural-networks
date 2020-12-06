import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from examples import Point
from examples import Examples
from nn import Neural_Network

arm_length = 100.0
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Open Sans', 24)
screen = pygame.display.set_mode([500, 500])
ex = Examples(arm_length)
e = ex.generate(1000)

def find_joints(angles):
    alpha = np.pi - angles[0]
    beta = angles[1]*(-1.0)
    first_joint = translate(Point(250.0, 250.0), alpha)
    second_joint = translate(first_joint, np.pi - beta + alpha)
    return first_joint, second_joint

def translate(center, angle):
    return Point(center.x + arm_length * np.sin(angle), center.y - arm_length * np.cos(angle))

def unstandarize(angles):
    return np.array(angles)*np.pi


def draw_range():
    for _ in e[0]:
        # print(f'pierwsze : {_[0]}')
        # print(f'drugie : {_[1]}')
        pygame.draw.circle(screen, (128,0,0), (_[0]*250+250, _[1]*250+250), 1.1)
        pygame.display.flip()


def main():

    # _tmp = translate(Point(0,0), np.pi/4.0)
    # print(_tmp.x)
    # print(_tmp.y)
    
    
    # print(e)
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # for (x, y) in e[0]:
    #     plt.scatter(x, y, marker='o')
    # plt.show()


    print(np.max(e[0]), np.max(e[1]))
    x_train = (np.array(e[0]) + arm_length*2)/(arm_length*4)*0.8 + 0.1
    # y_train = np.array(e[1])/np.pi
    y_train = np.array(e[1])/np.pi*0.8 + 0.1

    

    print(np.max(y_train))
    print(np.shape(e[0]))
    NN = Neural_Network()

    # NN.train(e[0][0])

    for i in range(100000):
        # index = np.random.randint(0, np.size(e[0], 0))
        NN.train(x_train, y_train)
    
    err = NN.errors
    plt.plot(range(len(err)), err)
    plt.show()

    screen.fill((255, 255, 255))
    image = pygame.image.load('robot.png')
    screen.blit(image, (0, 0))
    # draw_range()
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
                    x = pos[0] - 250
                    y = pos[1] - 250
                    # x /= 250
                    # y /= 250
                    # print(NN.forward((pos[0]-250, pos[1] - 250)))
                    # prediction = unstandarize(NN.forward((x, -y)))
                    prediction = unstandarize(NN.forward(((x + arm_length*2)/(arm_length*4)*0.8 + 0.1, (-y + arm_length*2)/(arm_length*4)*0.8 + 0.1)))
                    print(prediction)
                    joints = find_joints(prediction)
                    
                    print(joints[0].x)
                    print(joints[0].y)
                    print(joints[1].x)
                    print(joints[1].y)
                    screen.fill((255, 255, 255))
                    # draw_range()
                    image = pygame.image.load('robot.png')
                    screen.blit(image, (0, 0))
                    pygame.draw.line(screen, (255,0,0), (250, 250), (joints[0].x, joints[0].y), width=5)
                    pygame.draw.line(screen, (0,0,255), (joints[0].x, joints[0].y), (joints[1].x, joints[1].y), width=5)
                    pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()