modality: PET
scanner name: PET_Siemens_Quadra
description: Vision PET scanner by SIEMENS.
number of elements: 243200
number of layers: 1
voxels number transaxial: 220
voxels number axial: 220
field of view transaxial: 726
field of view axial: 261
scanner radius: 410
number of rsectors: 38
number of crystals transaxial: 5
number of crystals axial: 5 
crystals size depth: 20
crystals size trans: 3.2
crystals size axial: 3.2
rsectors first angle: 4.73684 # ? From Siemens Sinogram Organization file 7.10526 4.73684
                              # 1st bucket located on top, so first block is the 1st one with x positive

number of rsectors axial: 4
rsector gap axial: 3.29
number of modules transaxial: 2
number of modules axial: 8
module gap transaxial: 0.2
# D'après Mac file, gap =0.8
module gap axial: 0.8 #0.071428571
number of submodules transaxial: 2
number of submodules axial: 2 
submodule gap transaxial: 0.2
submodule gap axial: 0.2 
crystal gap transaxial: 0
crystal gap axial: 0
mean depth of interaction: 6.4 
# There is also the "LORDepthOfInteraction()=0.8cm" in the logs from SIEMENS, sinogramDepthOfInteraction() is 6.7mm, and effective detector ring radius=416.4mm (410mm+6.4mm)
min angle difference: 72.5 # ? Recovered from MCT geom file. Have to be defined from Siemens organization file with Vision sinogram size (should be higher than that)
                           # Anyway this parameter is not required since we will use normalization datafile for sensitivity generation
rotation direction: CW


# From recon logs:
# crystalsPerRing()=798 (with gaps)
# bucketsPerRing()=19
# transBlocks()=2
# transCrystals()=20
# transBlockGaps()=1


# Description from article doi:10.2967/jnumed.118.215418
#The PET component contains eight detector rings and 19 Detector Electronics Assembly (DEA)units to form a ring. 
#Two adjacent detector blocks per DEA results in 38 blocks per ring. 
#Each detector block contains a 4x2 arrangement of mini‐blocks. 
#A mini‐block consists of a 5x5 LSO‐array of 3.2x3.2x20 mm crystals coupled to a SiPM‐array. 
#Each SiPM‐array is 16x16 mm and has 16 output channels.
#The arrangement of 4x2 mini‐blocks, with two mini‐blocks in the axial direction, results in 32 mm axial FOV for one block. 
#This configuration, that uses eight blocks in the axial direction, has a 25.6 cm axial FOV, or 26.1 cm including the packing spaces between the blocks.
#The design of the detector is based on a square array of small crystals which area is fully covered by SiPM detector elements, exploiting the full potential of SiPMs. 
#The 3.2 mm crystal size allows for a high system spatial resolution, while the full coverage optimizes light collection and enables improved timing resolution and signal to noise ratio (13).

block
mini-block
crystals

