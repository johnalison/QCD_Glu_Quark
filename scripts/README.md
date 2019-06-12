
> py -i QCD_Glu_Quark/scripts/plotting.py --i test.parquet.0 -o TestPlots

where -i is a parquet file with events and -o is the output directory.

in the interactive session do:

>> drawEvent(3)  # to draw event number 3

or 

>> drawTest(3, X)  # to draw image number X for event number 3.  
                   # X = 0 for muons 
                   # X = 1 for tracks
                   # X = 2 for ECAL
                   # X = 3 for HCAL 




The following scripts have not yet been converted to run on parquet files:

#
# Plot the overall intestities to study the scale of the HCAL. 
#

 py QCD_Glu_Quark/scripts/plotIntensities.py --inputFileSig test_jets_BJetsAll.hdf5 --inputFileBkg test_jets_NonBJetsAll.hdf5 -o TestPlots

#
#  Plot the kinematics of signal and background{
#

 py QCD_Glu_Quark/scripts/plotkine.py

#
#  Plot sums/differences over many events. (Will try to do some not so smart caching) 
#

py QCD_Glu_Quark/scripts/plotsums.py --inputFileSig test_jets_BJetsAll.hdf5 --inputFileBkg test_jets_NonBJetsAll.hdf5 -o TestPlots/

