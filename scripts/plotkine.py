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

nevents = 10000
inputFileSigName = "test_jets_BJetsAll.hdf5"
inputFileSig =  h5py.File(inputFileSigName,"r")

for k in inputFileSig.keys():
    print("\t",k)

inputFileBkgName = "test_jets_NonBJetsAll.hdf5"
inputFileBkg =  h5py.File(inputFileBkgName,"r")


def makeplot(name,bins,xlabel,ylabel,log=False):

    sigHist = inputFileSig.get(name)
    sigHistData = sigHist[0:nevents]

    bkgHist = inputFileBkg.get(name)
    bkgHistData = bkgHist[0:nevents]


    fig, ax = plt.subplots(figsize=(6,5))
    n, bins, patches = plt.hist(sigHistData,bins=bins,color = "blue", linestyle='None',histtype='step', log=log)
    n, bins, patches = plt.hist(bkgHistData,bins=bins,color = "red",  linestyle='None',histtype='step', log=log)
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.savefig(name+".pdf")
    plt.close()

ptBins = np.linspace(20,120,21)
makeplot(name="jet_pT",xlabel=r'$p_T$ [GeV]',ylabel='Entries',bins=ptBins,log=True)

etaBins = np.linspace(-1.7,1.7,35)
makeplot(name="jet_eta",xlabel=r'$\eta$',ylabel='Entries',bins=etaBins)


phiBins = [-3.25,-3,-2.75,-2.5,-2.25,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25]
makeplot(name="jet_phi",xlabel=r'$\phi$',ylabel='Entries',bins=phiBins)

#jet_truthLabel
#
truthBins = np.linspace(-0.5,24.5,26)
print(truthBins)
makeplot(name="jet_truthLabel",xlabel=r'Truth Label',ylabel='Entries',bins=truthBins,log=True)



#import os
#if not os.path.isdir(o.outputDir):
#    print("making",o.outputDir)
#    os.mkdir(o.outputDir)
#
#
#nevents = 100000
#
#
#try:
#    print("Trying to Load: outputKine.hdf5" )
#    inputSumFile =  h5py.File("outputKine.hdf5","r")
#    X_jets_sig_sum = inputSumFile.get("X_jets_sig_sum")
#    X_jets_bkg_sum = inputSumFile.get("X_jets_bkg_sum")
#    X_jets_sig_sum = np.array(X_jets_sig_sum)
#    X_jets_bkg_sum = np.array(X_jets_bkg_sum)
#
#    inputSumFile.close()
#
#except:    
#    print("Creating the sums by hand" )
#    inputFileSig =  h5py.File(o.inputFileSig,"r")
#    inputFileBkg =  h5py.File(o.inputFileBkg,"r")
#
#    #hists = ["jetSeed_ieta","jetSeed_iphi","jet_eta","jet_pT","jet_phi","jet_truthLabel"]
#
#    #sigHists = []
#    #for h in hists:
#    sigHist = inputFileSig.get("jet_pT")
#
#    
#
#    #bkgHists = []
#    #for h in hists:
#    #    bkgHists.append(inputFileBkg.get(h))
#    #
#    #    
#    #
#    #X_jets_sig_subset = X_jets_sig[0:nevents]
#    #X_jets_sig_sum = np.sum(X_jets_sig_subset, axis=0)
#    #
#    #X_jets_bkg_subset = X_jets_bkg[0:nevents]
#    #X_jets_bkg_sum = np.sum(X_jets_bkg_subset, axis=0)
#    #
#    #outputFile = h5py.File("output.hdf5","w")
#    #outputFile.create_dataset('X_jets_sig_sum', data=X_jets_sig_sum)
#    #outputFile.create_dataset('X_jets_bkg_sum', data=X_jets_bkg_sum)
#    #outputFile.close()
#
#
#  
##plt.imshow(X_EB_data.reshape(X_EB_data.shape[0], matrix.shape[1]), cmap=plt.cm.Greys)
##plt.show()
#
#def drawSum(data,prefix,iImg,trkMax=10,caloScale=1,vmin=1e-3,alpha=1.0,figsize=(5.0,4.0)):
#    fig, ax = plt.subplots(figsize=figsize)
#    #fig, ax = plt.subplots()
#    #print("Size",fig.get_size_inches())
#    #fig, (ax, cax) = plt.subplots(ncols=2,figsize=(6,6), 
#    #                             gridspec_kw={"width_ratios":[1, 0.05]})
#    
#    # axis labeling
#    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
#    xmin = 0 
#    xmax = 125.+1.
#    #xmin = 32.5
#    #xmax = 92.5
#    ymin = xmin
#    ymax = xmax
#    
#    plt.xlim([xmin, xmax])
#    plt.xlabel(r'$\mathrm{i\varphi}$', size=14)
#    ax.xaxis.set_tick_params(direction='in', which='major', length=6.)
#    plt.ylim([ymin, ymax])
#    plt.ylabel(r'$\mathrm{i\eta}$', size=14)
#    ax.yaxis.set_tick_params(direction='in', which='major', length=6.)
#
#
#    #print(X_jets_subset.shape)    
#
#    #print(X_jets_sum.shape)
#    #
#    #print( X_jets_subset[0,:,:,1].shape)
#    #print( X_jets_sum[:,:,1].shape)
#
#    if iImg == 3:
#        im = ax.imshow(data[:,:,3],cmap="Greys",norm=LogNorm(),vmin=vmin, vmax=25*caloScale*nevents,alpha=alpha)
#        label = "HCAL"
#    elif iImg == 2: 
#        im = ax.imshow(data[:,:,2],cmap="Blues",norm=LogNorm(),vmin=vmin, vmax=caloScale*nevents   ,alpha=alpha)
#        label = "ECAL"
#    elif iImg == 1:
#        im = ax.imshow(data[:,:,1],cmap="Oranges", norm=LogNorm(),vmin=0.1, vmax=trkMax*nevents      ,alpha=alpha)
#        label = "Tracks"
#    elif iImg == 0:
#        im = ax.imshow(data[:,:,0],cmap="cool", norm=LogNorm(),vmin=1e-6, vmax=trkMax*nevents      ,alpha=alpha)
#        label = "Muons"
#
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="3%", pad=0.05)
#    cbar = fig.colorbar(im, cax=cax)
#    cbar.ax.set_ylabel(label+" Energy Sum [GeV]", rotation=-90,labelpad=15)
#
#    #
#    #caxHCal = divider.append_axes("right", size="3%", pad=0.05)
#    #cbarHCAl = fig.colorbar(imHCal, cax=caxHCal)
#    #cbarHCAl.ax.set_ylabel("HCal Energy [GeV]", rotation=-90)
#    #
#    #caxECal = divider.append_axes("right", size="3%", pad=0.75)
#    #cbarECal=fig.colorbar(imECal, cax=caxECal)
#    #cbarECal.ax.set_ylabel("ECal Energy [GeV]", rotation=-90)
#    #
#    #caxTrks = divider.append_axes("right", size="3%", pad=0.75)
#    #cbarTrks = fig.colorbar(imTrks, cax=caxTrks)
#    #cbarTrks.ax.set_ylabel("Tracker pT [GeV]", rotation=-90)
#
#    #caxMuons = divider.append_axes("right", size="3%", pad=0.75)
#    #cbarMuons = fig.colorbar(imMuons, cax=caxMuons)
#    #cbarMuons.ax.set_ylabel("Muon P_{T} [GeV]", rotation=-90)
#
#
#
#    plt.savefig(o.outputDir+"/"+prefix+str(label)+".pdf")
#    plt.close()
#
#
#def drawDiff(data1,data2,prefix,iImg,trkMax=10,caloScale=1,vmin=1e-3,alpha=1.0,figsize=(6.4,4.0)):
#    fig, ax = plt.subplots(figsize=figsize)
#    
#    # axis labeling
#    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
#    xmin = 0 
#    xmax = 125.+1.
#    ymin = xmin
#    ymax = xmax
#    
#    plt.xlim([xmin, xmax])
#    plt.xlabel(r'$\mathrm{i\varphi}$', size=14)
#    ax.xaxis.set_tick_params(direction='in', which='major', length=6.)
#    plt.ylim([ymin, ymax])
#    plt.ylabel(r'$\mathrm{i\eta}$', size=14)
#    ax.yaxis.set_tick_params(direction='in', which='major', length=6.)
#
#
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
#
#
#
#    plt.savefig(o.outputDir+"/"+prefix+str(label)+".pdf")
#    plt.close()
#
#
#
#drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",0)
#drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",1)
#drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",2)
#drawDiff(X_jets_sig_sum,X_jets_bkg_sum,"Diff_",3)
#
#
#drawSum(X_jets_sig_sum,"Sum_BJets_",0)
#drawSum(X_jets_sig_sum,"Sum_BJets_",1)
#drawSum(X_jets_sig_sum,"Sum_BJets_",2)
#drawSum(X_jets_sig_sum,"Sum_BJets_",3)
#
#drawSum(X_jets_bkg_sum,"Sum_NonBJets_",0)
#drawSum(X_jets_bkg_sum,"Sum_NonBJets_",1)
#drawSum(X_jets_bkg_sum,"Sum_NonBJets_",2)
#drawSum(X_jets_bkg_sum,"Sum_NonBJets_",3)
#



#drawSum(1)
#drawSum(2)
#drawSum(3)
#if o.savePlots:
#    for iEvent in range(nEvents):
#        print("\tSaving Event",iEvent)
#        drawEvent(iEvent,save=True)
#        #drawWholeEvent(iEvent,save=True)
