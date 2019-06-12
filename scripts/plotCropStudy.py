import numpy as np
import matplotlib.pyplot as plt

data = {}
data[125] = ( 0.685,  0.456,  0.867)
data[120] = ( 0.688,  0.453,  0.871)
data[110] = ( 0.686,  0.453,  0.869)
data[100] = ( 0.685,  0.457,  0.868)
data[90 ] = ( 0.683,  0.455,  0.867)
data[80 ] = ( 0.682,  0.454,  0.868)
data[70 ] = ( 0.675,  0.471,  0.863)
data[60 ] = ( 0.671,  0.477,  0.857)
data[50 ] = ( 0.657,  0.503,  0.846)
data[40 ] = ( 0.641,  0.528,  0.836)
data[30 ] = ( 0.600,  0.586,  0.795)
data[20 ] = ( 0.564,  0.633,  0.764)
data[10 ] = ( 0.528,  0.669,  0.730)


print(data.keys())

plt.axis([0, 125, 0, 1.1])
plt.xlabel("Crop Size")
plt.ylabel("Relative Performance")
#plt.plot(tpr,tpr,"r--")

cropSize = []
rocAUC   = []
rejFixEff = []
effFixRej = []

for k in data.keys():
    cropSize.append(k)
    #print(data[k][0]-0.5,data[125][0]-0.5)
    rocAUC   .append((data[k][0]-0.5)/(data[125][0]-0.5))
    #print(0.7-data[125][1],0.7-data[k][1])
    #print(0.7-data[125][1],0.7-data[k][1])
    rejFixEff.append((0.7-data[k][1])/(0.7-data[125][1]))

    effFixRej.append((data[k][2]-0.7)/(data[125][2]-0.7))

print(cropSize)
print(rocAUC)
plt.plot(cropSize,rocAUC,"k")
plt.plot(cropSize,rejFixEff,"r")
plt.plot(cropSize,effFixRej,"blue")
plt.savefig("CropStudy.pdf")
