'''
Create a class called Node which contains the corresponsing device info...each Node is stored in a global list of nodes
We use an adjacency list to store the graph 

Each node_id is linear from 1 to N and we need to register each node as we connect and store it to the globalStorage

Change from heap to array storage for the Node instances


'''
globalNodeStorage = {} #this is to be stored in the database and need to be updated
localNodeStorage = {}


def getNodeId():
    return 0

class Node:
    def __init__(self):
        self.node_data = {
            "node_id" : None,
            "_ip": None,
            "_mac": None,
            "neighbours": {}
        }
    
    def connectToNode(self, ip, mac):
        self.node_data["_ip"] = ip
        self.node_data["_mac"] = mac
        self.node_data["node_id"] = getNodeId()
    
    def addNeighbour(self, node_id, distance):
        self.node_data["neighbours"][node_id] = dist
    
'''
Problem: How to store the nodes in an array so that it can be retreived using a node Id
use HashMap()
'''



def getNodeData(node_id):
    nodeData = globalNodeStorage[node_id]
    return nodeData


def storeNodeToGlobalStorage(node: Node, globalNodeStorage):
    """
    Can be used to store and update to the global storage
    """
    nodeKey = node.node_data["node_id"]
    globalNodeStorage[nodeKey] = node.node_data

def storeNodeToLocalStorage(node: Node, localNodeStorage):
    
    nodeKey = node.node_data["node_id"]
    localNodeStorage[nodeKey] = node

def createRandomConnectedTopology():
    pass

#if we have a topology of the graphs, we need to update the Node values
def addEdgeConnection(node_id_1, node_id_2, dist, globalNodeStorage, localNodeStorage):
    
    #check if both ids are the same
    """
    Get the Node instances from the localNodeStorage and then mutate them to create new Nodes
    and Store them
    """
    if node_id_1==node_id_2:
        return
    
    node_1 = localNodeStorage[node_id_1]
    node_2 = localNodeStoragep[node_id_2]
    #mutate the nodes

    node_1.addNeighbour(node_id_2, dist)
    node_2.addNeighbour(node_id_1, dist)
    
    storeNodeToGlobalStorage(node_1, globalNodeStorage)
    storeNodeToLocalStorage(node_2, localNodeStorage)
    
    return

