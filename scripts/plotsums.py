import optparse
parser = optparse.OptionParser()
parser.add_option('-o', '--outputDir',           dest="outputDir",          default=False, help="Run in loop mode")
parser.add_option('--inputFileSig',           dest="inputFileSig",          default=False, help="Run in loop mode")
parser.add_option('--inputFileBkg',           dest="inputFileBkg",          default=False, help="Run in loop mode")
parser.add_option('-s', '--savePlots',           action="store_true", dest="savePlots",         default=False, help="")
o, a = parser.parse_args()

import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import h5py



import os
if not os.path.isdir(o.outputDir):
    print("making",o.outputDir)
    os.mkdir(o.outputDir)


nevents = 100000

try:
    print("Trying to Load: output.hdf5" )
    inputSumFile =  h5py.File("output.hdf5","r")
    X_jets_sig_sum = inputSumFile.get("X_jets_sig_sum")
    X_jets_bkg_sum = inputSumFile.get("X_jets_bkg_sum")
    X_jets_sig_sum = np.array(X_jets_sig_sum)
    X_jets_bkg_sum = np.array(X_jets_bkg_sum)

    inputSumFile.close()

except:    
    print("Creating the sums by hand" )
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

    X_jets_bkg_sum = np.sum(X_jets_bkg_subset_True, axis=0)
    X_jets_sig_sum = np.sum(X_jets_sig_subset_True, axis=0)

    #result = (jet_truthLabel = 5)
    #print(jet_truthLabelSig == 5)
    #bs = (jet_truthLabelSig == 5)
    #nbs = (jet_truthLabelSig != 5)
    #print(np.sum(bs))
    #print(np.sum(nbs))
    #print(X_jets_bkg.shape)
    #print(nbs.shape)

    #print(X_jets_bkg_True.shape)
    #for k in inputFileSig.keys():
    #    print(k)

    outputFile = h5py.File("output.hdf5","w")
    outputFile.create_dataset('X_jets_sig_sum', data=X_jets_sig_sum)
    outputFile.create_dataset('X_jets_bkg_sum', data=X_jets_bkg_sum)

    outputFile.close()


  
#plt.imshow(X_EB_data.reshape(X_EB_data.shape[0], matrix.shape[1]), cmap=plt.cm.Greys)
#plt.show()

def drawSum(data,prefix,iImg,trkMax=10,caloScale=1,vmin=1e-3,alpha=1.0,figsize=(5.0,4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    #fig, ax = plt.subplots()
    #print("Size",fig.get_size_inches())
    #fig, (ax, cax) = plt.subplots(ncols=2,figsize=(6,6), 
    #                             gridspec_kw={"width_ratios":[1, 0.05]})
    
    # axis labeling
    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
    xmin = 0 
    xmax = 125.+1.
    #xmin = 32.5
    #xmax = 92.5
    ymin = xmin
    ymax = xmax
    
    plt.xlim([xmin, xmax])
    plt.xlabel(r'$\mathrm{i\varphi}$', size=14)
    ax.xaxis.set_tick_params(direction='in', which='major', length=6.)
    plt.ylim([ymin, ymax])
    plt.ylabel(r'$\mathrm{i\eta}$', size=14)
    ax.yaxis.set_tick_params(direction='in', which='major', length=6.)


    #print(X_jets_subset.shape)    

    #print(X_jets_sum.shape)
    #
    #print( X_jets_subset[0,:,:,1].shape)
    #print( X_jets_sum[:,:,1].shape)

    if iImg == 3:
        im = ax.imshow(data[:,:,3],cmap="Greys",norm=LogNorm(),vmin=vmin, vmax=25*caloScale*nevents,alpha=alpha)
        label = "HCAL"
    elif iImg == 2: 
        im = ax.imshow(data[:,:,2],cmap="Blues",norm=LogNorm(),vmin=vmin, vmax=caloScale*nevents   ,alpha=alpha)
        label = "ECAL"
    elif iImg == 1:
        im = ax.imshow(data[:,:,1],cmap="Oranges", norm=LogNorm(),vmin=0.1, vmax=trkMax*nevents      ,alpha=alpha)
        label = "Tracks"
    elif iImg == 0:
        im = ax.imshow(data[:,:,0],cmap="cool", norm=LogNorm(),vmin=1e-6, vmax=trkMax*nevents      ,alpha=alpha)
        label = "Muons"

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(label+" Energy Sum [GeV]", rotation=-90,labelpad=15)

    #
    #caxHCal = divider.append_axes("right", size="3%", pad=0.05)
    #cbarHCAl = fig.colorbar(imHCal, cax=caxHCal)
    #cbarHCAl.ax.set_ylabel("HCal Energy [GeV]", rotation=-90)
    #
    #caxECal = divider.append_axes("right", size="3%", pad=0.75)
    #cbarECal=fig.colorbar(imECal, cax=caxECal)
    #cbarECal.ax.set_ylabel("ECal Energy [GeV]", rotation=-90)
    #
    #caxTrks = divider.append_axes("right", size="3%", pad=0.75)
    #cbarTrks = fig.colorbar(imTrks, cax=caxTrks)
    #cbarTrks.ax.set_ylabel("Tracker pT [GeV]", rotation=-90)

    #caxMuons = divider.append_axes("right", size="3%", pad=0.75)
    #cbarMuons = fig.colorbar(imMuons, cax=caxMuons)
    #cbarMuons.ax.set_ylabel("Muon P_{T} [GeV]", rotation=-90)



    plt.savefig(o.outputDir+"/"+prefix+str(label)+".pdf")
    plt.close()



def drawDiff(data1,data2,prefix,iImg,trkMax=10,caloScale=1,vmin=1e-3,alpha=1.0,figsize=(6.4,4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # axis labeling
    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
    xmin = 0 
    xmax = 125.+1.
    ymin = xmin
    ymax = xmax
    
    plt.xlim([xmin, xmax])
    plt.xlabel(r'$\mathrm{i\varphi}$', size=14)
    ax.xaxis.set_tick_params(direction='in', which='major', length=6.)
    plt.ylim([ymin, ymax])
    plt.ylabel(r'$\mathrm{i\eta}$', size=14)
    ax.yaxis.set_tick_params(direction='in', which='major', length=6.)

    cropSize = 80
    startCrop = 125//2-(cropSize//2)
    startCrop = 125//2-(cropSize//2)
    print(0,startCrop-1)
    
    if iImg == 3:
        diff = data1[:,:,3] - data2[:,:,3]
        caloScale = 1e3

        diff[0:startCrop-1,:] = 0
        diff[125-startCrop+1:125,:] = 0
        diff[:,125-startCrop+1:125] = 0
        diff[:,0:startCrop-1] = 0

        im = ax.imshow(diff,cmap="Greys",norm=SymLogNorm(linthresh=1,linscale=0.1,vmin=-(caloScale),vmax=caloScale),vmin=(-caloScale), vmax=caloScale,alpha=alpha)
        #im = ax.imshow(diff,cmap="Greys",vmin=-25*caloScale, vmax=25*caloScale,alpha=alpha)
        label = "HCAL"
    elif iImg == 2: 
        diff = data1[:,:,2] - data2[:,:,2]
        caloScale = 1e3

        diff[0:startCrop-1,:] = 0
        diff[125-startCrop+1:125,:] = 0
        diff[:,125-startCrop+1:125] = 0
        diff[:,0:startCrop-1] = 0

        im = ax.imshow(diff,cmap="Blues",norm=SymLogNorm(linthresh=1,linscale=0.1,vmin=-(caloScale),vmax=caloScale),vmin=(-caloScale), vmax=caloScale,alpha=alpha)
        #im = ax.imshow(diff,cmap="Blues",norm=LogNorm(),vmin=-caloScale*nevents, vmax=caloScale*nevents   ,alpha=alpha)
        label = "ECAL"
    elif iImg == 1:
        diff = data1[:,:,1] - data2[:,:,1]
        trkMax = 1e2

        diff[0:startCrop-1,:] = 0
        diff[125-startCrop+1:125,:] = 0
        diff[:,125-startCrop+1:125] = 0
        diff[:,0:startCrop-1] = 0


        im = ax.imshow(diff,cmap="Oranges",norm=SymLogNorm(linthresh=1,linscale=0.5,vmin=-(trkMax),vmax=trkMax),vmin=(-trkMax), vmax=trkMax,alpha=alpha)
        #im = ax.imshow(diff,cmap="Oranges", norm=LogNorm(),vmin=0.1, vmax=trkMax*nevents      ,alpha=alpha)
        label = "Tracks"
    elif iImg == 0:
        diff = data1[:,:,0] - data2[:,:,0]

        diff[0:startCrop-1,:] = 0
        diff[125-startCrop+1:125,:] = 0
        diff[:,125-startCrop+1:125] = 0
        diff[:,0:startCrop-1] = 0


        im = ax.imshow(diff,cmap="cool",norm=SymLogNorm(linthresh=1,linscale=0.5,vmin=-(trkMax),vmax=trkMax),vmin=(-trkMax), vmax=trkMax,alpha=alpha)
        #im = ax.imshow(diff,cmap="cool", norm=LogNorm(),vmin=1e-6, vmax=trkMax*nevents      ,alpha=alpha)
        label = "Muons"


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel(label+" Energy Difference (Sig-Bkg) [GeV]", rotation=-90,labelpad=15)



    plt.savefig(o.outputDir+"/"+prefix+str(label)+".pdf")
    plt.close()




def plotIntensities(dataSignal,dataBkg,figsize=(6.4,4.0)):#data1,data2,prefix,iImg,trkMax=10,caloScale=1,vmin=1e-3,alpha=1.0,
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
    #print(max(dataSignal[:,:,0].reshape(125*125)))
    #print(max(dataSignal[:,:,1].reshape(125*125)))
    #print(max(dataSignal[:,:,2].reshape(125*125)))
    #print(max(dataSignal[:,:,3].reshape(125*125)))
    bins = np.linspace(0,100,100+1)
    
    print(dataSignal[:,:,1].shape    )
    #ax.hist(dataSignal[:,:,0].reshape(125*125),bins=bins,histtype='step',label="Muons")
    #ax.hist(dataSignal[:,:,1].reshape(125*125),bins=bins,histtype='step',label="Tracks")
    ax.hist(dataSignal[:,:,1].flatten(),bins=bins,histtype='step',label="Tracks")
    #ax.hist(dataSignal[:,:,2].reshape(125*125),bins=bins,histtype='step',label="ECAL")
    #ax.hist(dataSignal[:,:,3].reshape(125*125),bins=bins,histtype='step',label="HCAL")
    #plt.legend(loc="best")

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



    plt.savefig(o.outputDir+"/PixelIntensities.pdf")
    plt.close()



drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",0)
drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",1)
drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",2)
drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",3)


drawSum(X_jets_sig_sum,"Sum_BJets_",0)
drawSum(X_jets_sig_sum,"Sum_BJets_",1)
drawSum(X_jets_sig_sum,"Sum_BJets_",2)
drawSum(X_jets_sig_sum,"Sum_BJets_",3)

drawSum(X_jets_bkg_sum,"Sum_NonBJets_",0)
drawSum(X_jets_bkg_sum,"Sum_NonBJets_",1)
drawSum(X_jets_bkg_sum,"Sum_NonBJets_",2)
drawSum(X_jets_bkg_sum,"Sum_NonBJets_",3)

#plotIntensities(X_jets_sig_subset,X_jets_bkg_subset)
#print(X_jets_bkg_sum[:,:,0].reshape(125*125))
#plt.hist(X_jets_bkg_sum[:,:,0].reshape(125*125))
#plt.show()

#drawSum(1)
#drawSum(2)
#drawSum(3)
#if o.savePlots:
#    for iEvent in range(nEvents):
#        print("\tSaving Event",iEvent)
#        drawEvent(iEvent,save=True)
#        #drawWholeEvent(iEvent,save=True)
