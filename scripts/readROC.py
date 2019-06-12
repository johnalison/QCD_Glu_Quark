import numpy as np
import matplotlib.pyplot as plt

import h5py

import optparse
parser = optparse.OptionParser()
parser.add_option('-o', '--outputFile',          default=False, help="Run in loop mode")
parser.add_option('-i', '--inputFile',           default=False, help="Run in loop mode")
#parser.add_option('-s', '--savePlots',           action="store_true", dest="savePlots",         default=False, help="")
o, a = parser.parse_args()


#inFileName = "BTagMetrics/METRICS/BJets_ResNet_blocks3_RH1o1_ECAL+HCAL+Trk_lr0.0005_gamma0.5every10ep_epochs30/metrics_epoch11_auc0.6688.hdf5"
inFileName = o.inputFile
h = h5py.File(inFileName,"r")
#for k in h.keys():
#    print("\t",k)

fpr = h.get("fpr")
tpr = h.get("tpr")

fpr = np.float32(fpr)
tpr = np.float32(tpr)
#print(fpr.shape)
#print(tpr.shape)

effAt70Eff = 0
iAt70Eff = 0
rejAt70Eff = 0

rejAt70Rej = 0
iAt70Rej = 0
effAt70Rej = 0

for i in range(tpr.shape[0]):
    if abs(tpr[i] - 0.7) < abs(effAt70Eff - 0.7):
        effAt70Eff = tpr[i]
        iAt70Eff = i

    if abs(fpr[i] - 0.7) < abs(rejAt70Rej - 0.7):
        rejAt70Rej = fpr[i]
        iAt70Rej = i

print("Eff70%=",tpr[iAt70Eff],"Rej70%=",fpr[iAt70Eff])
print("Eff Rej50%=",tpr[iAt70Rej],"Rej Rej50%=",fpr[iAt70Rej])


plt.axis([0, 1, 0, 1])
plt.ylabel("Btag Efficiency")
plt.xlabel("Non-b Efficiency")
plt.plot(tpr,tpr,"r--")



plt.plot((fpr[iAt70Eff],fpr[iAt70Eff]), (0,tpr[iAt70Eff]) , "k:")
plt.plot((0,fpr[iAt70Eff]), (tpr[iAt70Eff],tpr[iAt70Eff]) , "k:")

plt.plot((fpr[iAt70Rej],fpr[iAt70Rej]), (0,tpr[iAt70Rej]) , "k:")
plt.plot((0,fpr[iAt70Rej]), (tpr[iAt70Rej],tpr[iAt70Rej]) , "k:")


#plt.plot(fpr,tpr,"o")
plt.plot(fpr,tpr)
plt.plot(fpr[iAt70Eff], tpr[iAt70Eff], "ro")
plt.plot(fpr[iAt70Rej], tpr[iAt70Rej], "ro")
plt.savefig(o.outputFile)
#plt.show()

