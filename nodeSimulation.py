'''
This file Simulates the behaivior of a physical Node
Literature

Each Physical Node is a small Container with some parameters and functionalities

1. Memory: Each Node has some empty memory to store some values (we assume the memory to be a dictionary)
2. Functionalities: 
    1. Connect to the central Server
    2. Connect to other nodes in the same network
    3. Perform adaptive weight training
3. ROM
    1. mac
    2. ip
'''
import numpy as np

class PhysicalNode:
    def __init__(self, ip, mac):
        self._ip = ip
        self._mac = mac
        self._memory = {}
    
    def addToMemory(self, key, value):
        self._memory[key] = value
    
    