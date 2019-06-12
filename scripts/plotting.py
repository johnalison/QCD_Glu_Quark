
import optparse
parser = optparse.OptionParser()
parser.add_option('-o', '--outputDir',           dest="outputDir",          default=False, help="Run in loop mode")
parser.add_option('-i', '--inputFile',           dest="inputFile",          default=False, help="Run in loop mode")
parser.add_option('-s', '--savePlots',           action="store_true", dest="savePlots",         default=False, help="")
o, a = parser.parse_args()


import pyarrow.parquet as pq
import pyarrow as pa # pip install pyarrow==0.7.1
import numpy as np


import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py


pqIn = pq.ParquetFile(o.inputFile)
print(pqIn.metadata)
print(pqIn.schema)
#X = pqIn.read_row_group(0, columns=['y','pt','eta','phi','iphi','ieta','X_jet']).to_pydict()


#import sys
#sys.exit(-1)

###inputFile =  h5py.File(o.inputFile,"r")
###print(inputFile)
###for k in inputFile.keys():
###    print("\t",k)
###print(type(inputFile["X_EB"]))
###
####print(inputFile["X_EB"][()])
###
####key_list = inputFile.keys()
####for key in key_list:
####    print(key)


import os
if not os.path.isdir(o.outputDir):
    print("making",o.outputDir)
    os.mkdir(o.outputDir)


#X_jets = inputFile.get("X_jets")
#X_ECAL_stacked=inputFile.get("X_ECAL_stacked")
#
#jet_pT=inputFile.get("jet_pT")
#jet_truthLabel=inputFile.get("jet_truthLabel")
#jet_eta=inputFile.get("jet_eta")
#jet_phi=inputFile.get("jet_phi")
#jetSeed_iphi=inputFile.get("jetSeed_iphi")
#jetSeed_ieta=inputFile.get("jetSeed_ieta")
  
#plt.imshow(X_EB_data.reshape(X_EB_data.shape[0], matrix.shape[1]), cmap=plt.cm.Greys)
#plt.show()
#X = pqIn.read_row_group(0, columns=['y','pt','eta','phi','iphi','ieta','X_jet']).to_pydict()
#nEvents = X_jets.shape[0]

ys = pqIn.read(columns=['y']).to_pydict()['y']
#pq.read_table('example.parquet', columns=['one', 'three'])
nEvents = len(ys)

print("nEvents ", nEvents)

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_center_event(img,cropx,cropy):
    nImages,x,y = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:,startx:startx+cropx,starty:starty+cropy]


def drawEvent(iEvent,trkMax=10,caloScale=1,vmin=1e-3,save=False,alpha=1.0,figsize=(6.4,4.0),crop=125,plotName=""):
    fig, ax = plt.subplots(figsize=figsize)
    #fig, ax = plt.subplots()
    #print("Size",fig.get_size_inches())
    #fig, (ax, cax) = plt.subplots(ncols=2,figsize=(6,6), 
    #                             gridspec_kw={"width_ratios":[1, 0.05]})
    if crop > 125:
        crop = 125
    
    # axis labeling
    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
    xmin = 0 
    xmax = float(crop)+1.
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

    jetInfo = pqIn.read_row_group(iEvent, columns=['pt','eta','phi','y','iphi','ieta']).to_pydict()
    #print(jetInfo['pt'])
    thisPt = round(jetInfo['pt'][0],2)
    thisTruthLabel = int(jetInfo['y'][0])
    thisEta = round(jetInfo['eta'][0],2)
    thisPhi = round(jetInfo['phi'][0],2)
    plt.text(5, 120, "Jet pT/eta/phi = "+str(thisPt)+" GeV / "+str(thisEta)+" / "+str(thisPhi), fontsize=10, color='black', horizontalalignment="left")
    plt.text(5, 113, "Truth Label = "+str(thisTruthLabel), fontsize=10, color='black', horizontalalignment="left")
    thisSeed_iPhi = jetInfo['iphi'][0]
    thisSeed_iEta = jetInfo['ieta'][0]
    #print("Seed iEta/iPhi",thisSeed_iEta,thisSeed_iPhi)
    #print("scaled iEta/iPhi",(5*thisSeed_iEta)+2,(5*thisSeed_iPhi)+2)

    X_jets_thisEvent = pqIn.read_row_group(iEvent, columns=['X_jet']).to_pydict()['X_jet']
    X_jets_thisEvent = np.float32(X_jets_thisEvent[0])
    
    # X = pqIn.read_row_group(0, columns=['X_jet.list.item.list.item.list.item']).to_pydict()['X_jet'] # read row-by-row
    
    #X_jets_crop = crop_center_event(X_jets_[iEvent,:,:,:],crop,crop)
    print(X_jets_thisEvent.shape)
    X_jets_crop = crop_center_event(X_jets_thisEvent,crop,crop)

    #print(X_jets[iEvent].shape)
    print(X_jets_crop.shape)
    #nprint(X_jets[iEvent,:,:,3].shape)
    #print(HCAL_cropped.shape)

    imHCal = ax.imshow(X_jets_crop[3,:,:],cmap="Greys",norm=LogNorm(),vmin=vmin, vmax=25*caloScale,alpha=alpha)


    imECal = ax.imshow(X_jets_crop[2,:,:],cmap="Blues",norm=LogNorm(),vmin=vmin, vmax=caloScale   ,alpha=alpha)

    #imTrks = ax.imshow(X_jets_crop[1,:,:],cmap="PuOr",norm=SymLogNorm(linthresh=1,linscale=0.5,vmin=-(trkMax),vmax=trkMax),vmin=(-trkMax), vmax=trkMax,alpha=alpha)
    TracksAll = X_jets_crop[1,:,:]
    print("Sum All",np.sum(TracksAll))    

    TracksPositive = np.array(X_jets_crop[1,:,:])
    TracksPositive[TracksPositive < 0] = 0
    print("Sum Pos",np.sum(TracksPositive))    

    imTrksPos = ax.imshow(TracksPositive,cmap="Oranges", norm=LogNorm(),vmin=0.1, vmax=trkMax      ,alpha=alpha)


    TracksNegative = np.array(X_jets_crop[1,:,:])
    print("All check",np.sum(TracksNegative))        
    TracksNegative[TracksNegative > 0] = 0
    TracksNegative[TracksNegative < 0] *= -1
    print("Sum neg",np.sum(TracksNegative))        
    #print(max(TracksNegative))

    imTrksNeg = ax.imshow(TracksNegative,cmap="Oranges", norm=LogNorm(),vmin=0.1, vmax=trkMax      ,alpha=alpha)

    imMuons = ax.imshow(X_jets_crop[0,:,:],cmap="cool", norm=LogNorm(),vmin=1e-6, vmax=trkMax      ,alpha=alpha)
    #imTrks = ax.imshow(X_jets_crop[iEvent,:,:,0],cmap="RdYlGn",vmin=-trkMax, vmax=trkMax      ,alpha=alpha*0.4)


    divider = make_axes_locatable(ax)

    caxHCal = divider.append_axes("right", size="3%", pad=0.05)
    cbarHCAl = fig.colorbar(imHCal, cax=caxHCal)
    cbarHCAl.ax.set_ylabel("HCal Energy [GeV]", rotation=-90)

    caxECal = divider.append_axes("right", size="3%", pad=0.75)
    cbarECal=fig.colorbar(imECal, cax=caxECal)
    cbarECal.ax.set_ylabel("ECal Energy [GeV]", rotation=-90)

    caxTrks = divider.append_axes("right", size="3%", pad=0.75)
    cbarTrks = fig.colorbar(imTrksPos, cax=caxTrks)
    cbarTrks.ax.set_ylabel("Tracker pT [GeV]", rotation=-90)

    #caxTrksNeg = divider.append_axes("right", size="3%", pad=0.75)
    #cbarTrksNeg = fig.colorbar(imTrksNeg, cax=caxTrksNeg)
    #cbarTrksNeg.ax.set_ylabel("Tracker pT [GeV]", rotation=-90)


    #caxMuons = divider.append_axes("right", size="3%", pad=0.75)
    #cbarMuons = fig.colorbar(imMuons, cax=caxMuons)
    #cbarMuons.ax.set_ylabel("Muon P_{T} [GeV]", rotation=-90)



    if save:
        if plotName:
            plt.savefig(o.outputDir+"/"+plotName+".pdf")
        else:
            plt.savefig(o.outputDir+"/Event"+str(iEvent)+".pdf")

        plt.close()
    else:
        fig.tight_layout()
        plt.show()


def drawTest(iEvent,imgNum,trkMax=10,scale=1,vmin=1e-3,save=False,alpha=1.0,figsize=(6.4,4.0),crop=125,plotName=""):
    fig, ax = plt.subplots(figsize=figsize)
    #fig, ax = plt.subplots()
    #print("Size",fig.get_size_inches())
    #fig, (ax, cax) = plt.subplots(ncols=2,figsize=(6,6), 
    #                             gridspec_kw={"width_ratios":[1, 0.05]})
    if crop > 125:
        crop = 125
    
    # axis labeling
    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
    xmin = 0 
    xmax = float(crop)+1.
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

    jetInfo = pqIn.read_row_group(iEvent, columns=['pt','eta','phi','y','iphi','ieta']).to_pydict()
    #print(jetInfo['pt'])
    thisPt = round(jetInfo['pt'][0],2)
    thisTruthLabel = int(jetInfo['y'][0])
    thisEta = round(jetInfo['eta'][0],2)
    thisPhi = round(jetInfo['phi'][0],2)
    plt.text(5, 120, "Jet pT/eta/phi = "+str(thisPt)+" GeV / "+str(thisEta)+" / "+str(thisPhi), fontsize=10, color='black', horizontalalignment="left")
    plt.text(5, 113, "Truth Label = "+str(thisTruthLabel), fontsize=10, color='black', horizontalalignment="left")
    thisSeed_iPhi = jetInfo['iphi'][0]
    thisSeed_iEta = jetInfo['ieta'][0]
    #print("Seed iEta/iPhi",thisSeed_iEta,thisSeed_iPhi)
    #print("scaled iEta/iPhi",(5*thisSeed_iEta)+2,(5*thisSeed_iPhi)+2)

    X_jets_thisEvent = pqIn.read_row_group(iEvent, columns=['X_jet']).to_pydict()['X_jet']
    X_jets_thisEvent = np.float32(X_jets_thisEvent[0])
    
    # X = pqIn.read_row_group(0, columns=['X_jet.list.item.list.item.list.item']).to_pydict()['X_jet'] # read row-by-row
    
    #X_jets_crop = crop_center_event(X_jets_[iEvent,:,:,:],crop,crop)
    print(X_jets_thisEvent.shape)
    X_jets_crop = crop_center_event(X_jets_thisEvent,crop,crop)

    #print(X_jets[iEvent].shape)
    print(X_jets_crop.shape)
    #nprint(X_jets[iEvent,:,:,3].shape)
    #print(HCAL_cropped.shape)

    #imTrks = ax.imshow(X_jets_crop[1,:,:],cmap="PuOr",norm=SymLogNorm(linthresh=1,linscale=0.5,vmin=-(trkMax),vmax=trkMax),vmin=(-trkMax), vmax=trkMax,alpha=alpha)
    ImgAll = X_jets_crop[imgNum,:,:]
    print("Sum All",np.sum(ImgAll))    

    ImgPositive = np.array(X_jets_crop[imgNum,:,:])
    ImgPositive[ImgPositive < 0] = 0
    print("Sum Pos",np.sum(ImgPositive))    

    imPos = ax.imshow(ImgPositive,cmap="Oranges", norm=LogNorm(),vmin=vmin, vmax=trkMax      ,alpha=alpha)


    ImgNegative = np.array(X_jets_crop[imgNum,:,:])
    print("All check",np.sum(ImgNegative))        
    ImgNegative[ImgNegative > 0] = 0
    ImgNegative[ImgNegative < 0] *= -1
    print("Sum neg",np.sum(ImgNegative))        
    #print(max(TracksNegative))

    imNeg = ax.imshow(ImgNegative,cmap="Oranges", norm=LogNorm(),vmin=vmin, vmax=trkMax      ,alpha=alpha)

    divider = make_axes_locatable(ax)

    caxTrks = divider.append_axes("right", size="3%", pad=0.75)
    cbarTrks = fig.colorbar(imPos, cax=caxTrks)
    #cbarTrks = fig.colorbar(imNeg, cax=caxTrks)
    cbarTrks.ax.set_ylabel("Test", rotation=-90)


    if save:
        if plotName:
            plt.savefig(o.outputDir+"/Test"+plotName+".pdf")
        else:
            plt.savefig(o.outputDir+"/TestEvent"+str(iEvent)+".pdf")

        plt.close()
    else:
        fig.tight_layout()
        plt.show()





def drawWholeEvent(iEvent,trkMax=10,caloScale=1,vmin=1e-3,save=False,alpha=1.0):
    fig, ax = plt.subplots(figsize=(10,6))

    #fig, (ax, cax) = plt.subplots(ncols=2,figsize=(6,6), 
    #                             gridspec_kw={"width_ratios":[1, 0.05]})
    
    # axis labeling
    # Note: due to the way imshow() renders images by default, the below lines will appear to 'flip' the image
    #xmin = 0 
    #xmax = 125.+1.
    plt.xlabel(r'$\mathrm{i\varphi}$', size=14)
    ax.xaxis.set_tick_params(direction='in', which='major', length=6.)
    plt.ylabel(r'$\mathrm{i\eta}$', size=14)
    ax.yaxis.set_tick_params(direction='in', which='major', length=6.)

    imHCal = ax.imshow(X_ECAL_stacked[iEvent,:,:,3],cmap="Greys",norm=LogNorm(),vmin=vmin, vmax=25*caloScale,alpha=alpha)


    imECal = ax.imshow(X_ECAL_stacked[iEvent,:,:,2],cmap="Blues",norm=LogNorm(),vmin=vmin, vmax=caloScale   ,alpha=alpha)


    imTrks = ax.imshow(X_ECAL_stacked[iEvent,:,:,1],cmap="Oranges", norm=LogNorm(),vmin=0.1, vmax=trkMax      ,alpha=alpha)
    imMuons = ax.imshow(X_ECAL_stacked[iEvent,:,:,0],cmap="cool", norm=LogNorm(),vmin=1e-6, vmax=trkMax      ,alpha=alpha)
    #imTrks = ax.imshow(X_jets[iEvent,:,:,0],cmap="RdYlGn",vmin=-trkMax, vmax=trkMax      ,alpha=alpha*0.4)
    plt.plot([0,359],[54.5,54.5],  "k--",linewidth=0.75,alpha=0.5)
    #line1.set_alpha(0.2)
    #line1.set_linewidth(0.5)

    plt.plot([0,359],[224.5,224.5],"k--",linewidth=0.75,alpha=0.5)

    divider = make_axes_locatable(ax)

    caxHCal = divider.append_axes("right", size="3%", pad=0.05)
    cbarHCAl = fig.colorbar(imHCal, cax=caxHCal)
    cbarHCAl.ax.set_ylabel("HCal Energy [GeV]", rotation=-90)

    caxECal = divider.append_axes("right", size="3%", pad=0.75)
    cbarECal=fig.colorbar(imECal, cax=caxECal)
    cbarECal.ax.set_ylabel("ECal Energy [GeV]", rotation=-90)

    caxTrks = divider.append_axes("right", size="3%", pad=0.75)
    cbarTrks = fig.colorbar(imTrks, cax=caxTrks)
    cbarTrks.ax.set_ylabel("Tracker P_T [GeV]", rotation=-90)

    #caxMuons = divider.append_axes("right", size="3%", pad=0.75)
    #cbarMuons = fig.colorbar(imMuons, cax=caxMuons)
    #cbarMuons.ax.set_ylabel("Muon P_{T} [GeV]", rotation=-90)




    if save:
        plt.savefig(o.outputDir+"/WholeEvent"+str(iEvent)+".pdf")
        plt.close()
    else:
        fig.tight_layout()
        plt.show()





if o.savePlots:
    for iEvent in range(nEvents):
        print("\tSaving Event",iEvent)
        drawEvent(iEvent,save=True)
        #drawWholeEvent(iEvent,save=True)
