# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:41:21 2022

@author: 38628
"""

import numpy as np
import matplotlib.pyplot as plt

#
# fig = plt.figure(figsize=(6,2))
# plt.axis('equal')
# ax = fig.add_subplot(111)
# ax.spines['left'].set_color('none')
# ax.spines['bottom'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.set_xticks([])
# ax.set_yticks([])
#
# xy = [[-1.5,-0.5],[-0.5,-0.5],[0.5,-0.5],[1.5,-0.5],
#       [-1.5,0.5],[-0.5,0.5],[0.5,0.5],[1.5,0.5]]
# theta = np.arange(0, 2 * np.pi,2 * np.pi / 100000)
# for i in range(len(xy)):
#     x,y = xy[i]
#     r = 0.2
#     x = x + r*np.cos(theta)
#     y = y + r*np.sin(theta)
#     plt.plot(x, y, color='black')
#
# theta = np.arange(-np.pi/2, np.pi/2 , 2 * np.pi / 100000)
# x,y = -3.0, 0.0
# r = 1.0
# x = x + r*np.cos(theta)
# y = y + r*np.sin(theta)
# plt.plot(x, y, color='black')
#
# theta = np.arange(np.pi/2, 3*np.pi/2 , 2 * np.pi / 100000)
# x,y = 3.0, 0.0
# r = 1.0
# x = x + r*np.cos(theta)
# y = y + r*np.sin(theta)
# plt.plot(x, y, color='black')
#
# plt.plot([-3,3], [1,1], linewidth=1.5, color='black')
# plt.plot([-3,3], [-1,-1], linewidth=1.5, color='black')
#
# plt.savefig('./complex_domain_2.pdf')
# plt.show()


# read 2dElasticity_Ⅱ_r.npy and 2dElasticity_Ⅱ_xy.npy
r = np.load('2dElasticity_Ⅲ_r.npy')
xy = np.load('2dElasticity_Ⅲ_xy.npy')

print(xy)
print(r)
