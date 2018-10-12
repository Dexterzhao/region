from region.max_p_regions.heuristics import MaxPRegionsHeu
from libpysal.io.fileio import FileIO as psopen
from libpysal.weights import Rook, Queen
from region.p_regions.azp import AZPTabu
import numpy

import geopandas as gp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def constructRegions(attr, spatially_extensive_attr, weight, metric, spatialThre, max_p):
    # This list will hold the final cluster assignment for each area.
    # There are two reserved values:
    #    -1 - Indicates an enclave
    #     0 - Means the area hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [0]*len(spatially_extensive_attr)

    # C is the ID of the current cluster.    
    C = 0
    regionSpatialAttr = {}
    enclave = []
    regionList = {}
    # This outer loop is just responsible for picking new seed areas--an area
    # from which to grow a new cluster.
    # Once a valid seed area is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each area P in the Dataset...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)

    for P in range(0, len(spatially_extensive_attr)):
    
        # Only areas that have not already been claimed can be picked as new 
        # seed areas.    
        # If the area's label is not 0, continue to the next area.
        if not (labels[P] == 0):
            continue
        
        # Find all of P's neighboring areas.
        NeighborPolys = weight.neighbors[P]
        
        # If the number is below 0, this area is an island. 
        if len(NeighborPolys) < 0:
            labels[P] = -1
        # Otherwise, if there are at least MinPolys nearby, use this area as the 
        # seed for a new cluster.    
        else: 
            C += 1
            labeledID, spatialAttrTotal = growClusterForPoly(labels, spatially_extensive_attr, P, NeighborPolys, C, weight, spatialThre)

            regionSpatialAttr[C] = spatialAttrTotal
            
            if spatialAttrTotal < spatialThre:
                enclave.extend(labeledID)
            else:
                regionList[C] = labeledID
            
    print(regionSpatialAttr) 
    
    num_regions = len(regionList)
    
    print(num_regions) 
    for i, l in enumerate(labels):
        if l == -1:
            enclave.append(i)
    print(enclave)
    print(len(enclave))
    
    if num_regions <= max_p:
        return None
    else:
        distanceMatrix = pdist(attr, metric=metric)
        distanceMatrix = squareform(distanceMatrix)
        
        assignEnclave(enclave, labels, regionList, weight, distanceMatrix)
        print(enclave)
        print(regionList)
        
        totalWithinRegionDistance = calculateWithinRegionDistance(regionList, distanceMatrix)
        print(totalWithinRegionDistance)

    # All data has been clustered!
    return labels, regionList, regionSpatialAttr, enclave

def growClusterForPoly(labels, spatially_extensive_attr, P, NeighborPolys, C, weight, spatialThre):
    """
    Grow a new cluster with label `C` from the seed area `P`.
    
    This function searches through the dataset to find all areas that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      ``      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset areas
      `P`      - Index of the seed area for this new cluster
      `NeighborPolys` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPolys` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed area.
    labels[P] = C
    labeledID = [P]
    spatialAttrTotal = spatially_extensive_attr[P]
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPolys will be used as a FIFO queue of areas to search--that is, it
    # will grow as we discover new branch areas for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPolys, the areas are represented by their index in the original
    # dataset.
    i = 0
    
    while i < len(NeighborPolys):    
        
        if spatialAttrTotal >= spatialThre:
            i += 1
            continue
        
        # Get the next area from the queue.        
        Pn = NeighborPolys[i]
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        if labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C
            labeledID.append(Pn)
            spatialAttrTotal += spatially_extensive_attr[Pn]
            # Find all the neighbors of Pn
            if spatialAttrTotal < spatialThre:                
                PnNeighborPolys = weight.neighbors[Pn]
                NeighborPolys = NeighborPolys + PnNeighborPolys            
        i += 1        
    return labeledID, spatialAttrTotal
    # We've finished growing cluster C!
    
def assignEnclave(enclave, labels, regionList, weight, distanceMatrix):    
    while len(enclave) > 0:
        ec = enclave[0]
        ecNeighbors = weight.neighbors[ec]
        minDistance = numpy.Inf
        assignedRegion = 0
        for ecn in ecNeighbors:            
            if ecn in enclave:
                continue
            rm = numpy.array(regionList[labels[ecn]])
            totalDistance = distanceMatrix[ecn, rm].sum()
            if totalDistance < minDistance:
                minDistance = totalDistance
                assignedRegion = labels[ecn]
                #print(minDistance)
        if assignedRegion == 0:
            print("Island")
        else:
            labels[ec] = assignedRegion
            regionList[assignedRegion].append(ec) 
            del enclave[0]
            
def calculateWithinRegionDistance(regionList, distanceMatrix):
    totalWithinRegionDistance = 0
    
    for k, v in regionList.items():  
        nv = numpy.array(v)
        regionDistance = 0
        for vv in v:
            regionDistance += distanceMatrix[vv, nv].sum()
        regionDistance = regionDistance/2
        
        totalWithinRegionDistance += regionDistance
        
    return totalWithinRegionDistance
        
    
        
if __name__ != 'main':
    #test_w_basic()
    dbfReader = psopen('data\\n529.dbf')
    attr = dbfReader.by_col('SAR1')
    attr = numpy.reshape(numpy.array(attr), (-1,1))
    spatially_extensive_attr = numpy.array(dbfReader.by_col('Uniform2'))
    print(spatially_extensive_attr.sum())
    w = Rook.from_shapefile('data\\n529.shp')
    
    shuffleTimes = 4
    max_p = 0
    for st in range(shuffleTimes):
        arr = numpy.arange(attr.size)
        numpy.random.shuffle(arr)
        
        attr = attr[arr]
        spatially_extensive_attr = spatially_extensive_attr[arr]
        #print(attr)
        scanResults = constructRegions(attr, spatially_extensive_attr, w, 'cityblock', 100, max_p) 
        #print(scanResults)
        
        if scanResults:
            max_p = len(scanResults[1])
            gp_shp = gp.read_file('data\\n529.shp')
            gp_shp['regions'] = scanResults[0]
            gp_shp.plot(column='regions', legend = True)
            plt.show()    
    
    

