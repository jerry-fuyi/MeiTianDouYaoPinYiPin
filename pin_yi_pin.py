#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
from scipy.signal import correlate2d

import sys
import datetime

import matplotlib.pyplot as plt


# In[2]:


def board_create(month, date, day, empty=False):
    board = np.zeros((8, 7), dtype=np.int32)
    board[0:2, 6] = 1
    board[7, 0:4] = 1
    if empty:
        return board
    
    if month<1 or month>12:
        return
    if date<1 or date>31:
        return
    if day<1 or day>7:
        return
    
    specials = []
    
    i = (month-1)//6
    j = (month-1)%6
    board[i, j] = 1
    specials.append([i, j])
    
    i = (date-1)//7+2
    j = (date-1)%7
    board[i, j] = 1
    specials.append([i, j])
    
    if day <= 4:
        i = 6
        j = day+2
    else:
        i = 7
        j = day-1
    board[i, j] = 1
    specials.append([i, j])
    
    return board, specials


# In[35]:


def error():
    print("Invalid input!")
    exit(1)


# In[36]:


if len(sys.argv) == 4:
    try:
        month = int(sys.argv[1])
        date = int(sys.argv[2])
        day = int(sys.argv[3])
    except ValueError:
        error()
elif len(sys.argv) == 1:
    today = datetime.date.today()
    month = int(today.strftime("%m"))
    date = int(today.strftime("%d"))
    day = today.weekday() + 1
else:
    error()


# In[27]:


board = board_create(month, date, day)
if board is None:
    error()
board, specials = board


# In[4]:


class Chip:
    def __init__(self, shape):
        self.shape = shape.astype(np.int32)
        
        self.shapes = [shape]
        
        temp = []
        for s in self.shapes:
            ns = np.rot90(s)
            if self.can_add_to_shapes(ns):
                temp.append(ns)
        self.shapes += temp
        
        temp = []
        for s in self.shapes:
            ns = np.flip(s)
            if self.can_add_to_shapes(ns):
                temp.append(ns)
        self.shapes += temp
        
        temp = []
        for s in self.shapes:
            ns = np.flip(s, axis=0)
            if self.can_add_to_shapes(ns):
                temp.append(ns)
        self.shapes += temp
        
    def can_add_to_shapes(self, new_shape):
        for s in self.shapes:
            if np.array_equal(s, new_shape):
                return False
        return True
        
    def get_shapes(self):
        return self.shapes
    
    def get_weight(self):
        return np.sum(self.shape)


# In[5]:


chips = [
    Chip(np.array([[1, 1, 1, 1]])),
    Chip(np.array([[1, 1, 1],
                   [1, 0, 1]])),
    Chip(np.array([[1, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1]])),
    Chip(np.array([[0, 1, 0],
                   [0, 1, 0],
                   [1, 1, 1]])),
    Chip(np.array([[1, 1, 0],
                   [0, 1, 1]])),
    Chip(np.array([[1, 1, 0],
                   [0, 1, 0],
                   [0, 1, 1]])),
    Chip(np.array([[1, 1, 1],
                   [1, 0, 0]])),
    Chip(np.array([[1, 1, 1],
                   [1, 1, 0]])),
    Chip(np.array([[1, 1, 1, 1],
                   [1, 0, 0, 0]])),
    Chip(np.array([[1, 1, 1, 0],
                   [0, 0, 1, 1]]))
]


# In[6]:


required_min_adj = [board.size] * (len(chips)-1)

for i in range(len(chips)):
    for j in range(i):
        required_min_adj[j] = min(required_min_adj[j], chips[i].get_weight())
        
required_min_adj.append(0)


# In[7]:


def min_adjacent(board, target):
    board = board.copy()
    min_value = board.size
    
    for i0 in range(board.shape[0]):
        for j0 in range(board.shape[1]):
            if board[i0, j0] == target:
                count = 0
                ls = [[i0, j0]]
                
                while len(ls) != 0:
                    i, j = ls[0]
                    ls.pop(0)
                    board[i, j] = 1 - target
                    count += 1
                    if i != 0 and board[i-1, j] == target:
                        ls.append([i-1, j])
                    if i != board.shape[0]-1 and board[i+1, j] == target:
                        ls.append([i+1, j])
                    if j != 0 and board[i, j-1] == target:
                        ls.append([i, j-1])
                    if j != board.shape[1]-1 and board[i, j+1] == target:
                        ls.append([i, j+1])
                
                min_value = min(min_value, count)
    
    return min_value


# In[8]:


ok = False
info = [0] * len(chips)

def dfs(idx, board_cur):
    global ok
    
    if idx == len(chips):
        ok = True
        return
    
    for c in chips[idx].get_shapes():
        cor = correlate2d(board_cur, c, mode='valid')
        for i in range(cor.shape[0]):
            for j in range(cor.shape[1]):
                if cor[i, j] == 0:
                    board_new = board_cur.copy()
                    board_new[i:i+c.shape[0], j:j+c.shape[1]] += c
                    if min_adjacent(board_new, target=0) < required_min_adj[idx]:
                        break
                    info[idx] = [c, i, j]
                    dfs(idx+1, board_new)
                    if ok:
                        return


# In[9]:


dfs(0, board)


# In[10]:


def get_boundary(shape, target, y0=0, x0=0):
    s = shape.shape
    hor = np.zeros((s[0]+1, s[1]))
    ver = np.zeros((s[0], s[1]+1))
    for i in range(s[0]):
        for j in range(s[1]):
            if shape[i, j] == target:
                hor[i, j] = 1 - hor[i, j]
                hor[i+1, j] = 1 - hor[i+1, j]
                ver[i, j] = 1 - ver[i, j]
                ver[i, j+1] = 1 - ver[i, j+1]
                
    result = []
    for i in range(s[0]+1):
        for j in range(s[1]):
            if hor[i, j] == 1:
                result.append([[x0+j, x0+j+1], [y0+i, y0+i]])
    for i in range(s[0]):
        for j in range(s[1]+1):
            if ver[i, j] == 1:
                result.append([[x0+j, x0+j], [y0+i, y0+i+1]])
    
    return result


# In[ ]:


plt.figure(figsize=(6, 6))

for shape, y0, x0 in info:
    boundary = get_boundary(shape, 1, y0, x0)
    for lx, ly in boundary:
        plt.plot(lx, ly, c='C0')
        
for y, x in specials:
    plt.fill_between([x, x+1], [y, y], [y+1, y+1], color='C2')

boundary = get_boundary(board_create(0, 0, 0, True), 0, 0, 0)
for lx, ly in boundary:
    plt.plot(lx, ly, c='C1')

plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# In[ ]:




