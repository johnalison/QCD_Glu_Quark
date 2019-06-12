import optparse
parser = optparse.OptionParser()
parser.add_option('-o', '--outputDir',           dest="outputDir",          default=False, help="Run in loop mode")
parser.add_option('--inputFileSig',           dest="inputFileSig",          default=False, help="Run in loop mode")
parser.add_option('--inputFileBkg',           dest="inputFileBkg",          default=False, help="Run in loop mode")
o, a = parser.parse_args()


import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import h5py


nevents = 1000

inputFileSig =  h5py.File(o.inputFileSig,"r")

X_jets_sig = inputFileSig.get("X_jets")
nEventsSig = X_jets_sig.shape[0]
print("nEventsSig ", nEventsSig)
assert(nevents < nEventsSig)


inputFileBkg =  h5py.File(o.inputFileBkg,"r")
X_jets_bkg = inputFileBkg.get("X_jets")
nEventsBkg = X_jets_bkg.shape[0]
print("nEventsBkg ", nEventsBkg)
assert(nevents < nEventsBkg)

jet_truthLabelSig = inputFileSig.get("jet_truthLabel")
jet_truthLabelSig = np.array(jet_truthLabelSig)
jet_truthLabelSig_subset = jet_truthLabelSig[0:nevents]


X_jets_sig_subset_Raw = X_jets_sig[0:nevents]
X_jets_sig_subset_True = X_jets_sig_subset_Raw[jet_truthLabelSig_subset==5,:,:,:]


print(X_jets_sig_subset_Raw.shape,"--->",X_jets_sig_subset_True.shape)


jet_truthLabelBkg = inputFileBkg.get("jet_truthLabel")
jet_truthLabelBkg = np.array(jet_truthLabelBkg)
jet_truthLabelBkg_subset = jet_truthLabelBkg[0:nevents]


X_jets_bkg_subset_Raw = X_jets_bkg[0:nevents]
X_jets_bkg_subset_True = X_jets_bkg_subset_Raw[jet_truthLabelBkg_subset!=5,:,:,:]


print(X_jets_bkg_subset_Raw.shape,"--->",X_jets_bkg_subset_True.shape)


#
# Recaling to min
#
nEventsAfterTruthCut = min(X_jets_bkg_subset_True.shape[0],X_jets_sig_subset_True.shape[0])
print(nEventsAfterTruthCut)

X_jets_bkg_subset_True = X_jets_bkg_subset_True[0:nEventsAfterTruthCut]
X_jets_sig_subset_True = X_jets_sig_subset_True[0:nEventsAfterTruthCut]
print(X_jets_bkg_subset_True.shape,"and",X_jets_sig_subset_True.shape)


def flattenAndZeroSuppres(dataIn):
    dataOut = dataIn.flatten()
    dataOut = dataOut[dataOut != 0]
    return dataOut

def plotIntensities(dataSignal,dataBkg,HCALScale=1,figsize=(6.4,4.0)):#data1,data2,prefix,iImg,trkMax=10,caloScale=1,vmin=1e-3,alpha=1.0,
    fig, ax = plt.subplots(figsize=figsize)

    
    # axis labeling
    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
    #xmin = 0 
    #xmax = 125.+1.
    #ymin = xmin
    #ymax = xmax
    
    #plt.xlim([xmin, xmax])
    #plt.xlabel(r'$\mathrm{i\varphi}$', size=14)
    #ax.xaxis.set_tick_params(direction='in', which='major', length=6.)
    #plt.ylim([ymin, ymax])
    #plt.ylabel(r'$\mathrm{i\eta}$', size=14)
    #ax.yaxis.set_tick_params(direction='in', which='major', length=6.)
    #print(max(dataSignal[:,:,:,0].flatten()))
    #print(max(dataSignal[:,:,:,1].flatten()))
    #print(max(dataSignal[:,:,:,2].flatten()))
    #print(max(dataSignal[:,:,:,3].flatten()))
    bins = np.linspace(0,50,100+1)
    
    print("Signal Size",dataSignal[:,:,:,1].shape    )
    #print("Signal Size",dataSignal[:,:,:,1].flatten().shape    )

    
    ax.hist(flattenAndZeroSuppres(dataSignal[:,:,:,0]),bins=bins,histtype='step',label="Muons",color="red")
    ax.hist(flattenAndZeroSuppres(dataSignal[:,:,:,1]),bins=bins,histtype='step',label="Tracks",color="orange")
    ax.hist(flattenAndZeroSuppres(dataSignal[:,:,:,2])*1,bins=bins,histtype='step',label="ECAL",color="blue")
    ax.hist(flattenAndZeroSuppres(dataSignal[:,:,:,3])*HCALScale,bins=bins,histtype='step',label="HCAL",color="black")

#    ax.hist(flattenAndZeroSuppres(dataBkg[:,:,:,0]),bins=bins,histtype='step',label="Muons",color="red",linestyle=('dashed'))#))
#    ax.hist(flattenAndZeroSuppres(dataBkg[:,:,:,1]),bins=bins,histtype='step',label="Tracks",color="orange",linestyle=('dashed'))
#    ax.hist(flattenAndZeroSuppres(dataBkg[:,:,:,2]),bins=bins,histtype='step',label="ECAL",color="blue",linestyle=('dashed'))
#    ax.hist(flattenAndZeroSuppres(dataBkg[:,:,:,3]),bins=bins,histtype='step',label="HCAL",color="black",linestyle=('dashed'))

    plt.legend(loc="best")

    ax.set_yscale('log')

#    if iImg == 3:
#        diff = data1[:,:,3] - data2[:,:,3]
#        caloScale = 1e3
#        im = ax.imshow(diff,cmap="Greys",norm=SymLogNorm(linthresh=1,linscale=0.1,vmin=-(caloScale),vmax=caloScale),vmin=(-caloScale), vmax=caloScale,alpha=alpha)
#        #im = ax.imshow(diff,cmap="Greys",vmin=-25*caloScale, vmax=25*caloScale,alpha=alpha)
#        label = "HCAL"
#    elif iImg == 2: 
#        diff = data1[:,:,2] - data2[:,:,2]
#        caloScale = 1e3
#        im = ax.imshow(diff,cmap="Blues",norm=SymLogNorm(linthresh=1,linscale=0.1,vmin=-(caloScale),vmax=caloScale),vmin=(-caloScale), vmax=caloScale,alpha=alpha)
#        #im = ax.imshow(diff,cmap="Blues",norm=LogNorm(),vmin=-caloScale*nevents, vmax=caloScale*nevents   ,alpha=alpha)
#        label = "ECAL"
#    elif iImg == 1:
#        diff = data1[:,:,1] - data2[:,:,1]
#        trkMax = 1e2
#        im = ax.imshow(diff,cmap="Oranges",norm=SymLogNorm(linthresh=1,linscale=0.5,vmin=-(trkMax),vmax=trkMax),vmin=(-trkMax), vmax=trkMax,alpha=alpha)
#        #im = ax.imshow(diff,cmap="Oranges", norm=LogNorm(),vmin=0.1, vmax=trkMax*nevents      ,alpha=alpha)
#        label = "Tracks"
#    elif iImg == 0:
#        diff = data1[:,:,0] - data2[:,:,0]
#        im = ax.imshow(diff,cmap="cool",norm=SymLogNorm(linthresh=1,linscale=0.5,vmin=-(trkMax),vmax=trkMax),vmin=(-trkMax), vmax=trkMax,alpha=alpha)
#        #im = ax.imshow(diff,cmap="cool", norm=LogNorm(),vmin=1e-6, vmax=trkMax*nevents      ,alpha=alpha)
#        label = "Muons"
#
#
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="3%", pad=0.05)
#    cbar = fig.colorbar(im, cax=cax)
#    cbar.ax.set_ylabel(label+" Energy Difference (Sig-Bkg) [GeV]", rotation=-90,labelpad=15)



    plt.savefig(o.outputDir+"/PixelIntensities_HCALScale"+str(HCALScale)+".pdf")
    plt.close()


#plotIntensities(
plotIntensities(X_jets_sig_subset_True,X_jets_bkg_subset_True,HCALScale=0.25)
plotIntensities(X_jets_sig_subset_True,X_jets_bkg_subset_True,HCALScale=1)
plotIntensities(X_jets_sig_subset_True,X_jets_bkg_subset_True,HCALScale=25)
