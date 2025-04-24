# comments
#       Y                                        _________  
#       |                                       / _ \     \ 
#       |                                      | / \ |     |
#       |_____ Z                               | | | |     |
#        \                                     | | | |     |
#         \                                    | \_/ |     |
#          X                                    \___/_____/
# Left-handed axis orientation
# scanner axis is z
# positions in millimeters
# Use comma without space as separator in the tables.

# MANDATORY FIELDS
modality : PET
scanner name : GE_geometry_layer
number of elements              : 20160
number of layers : 1

#voxels number transaxial        : 9
#voxels number axial                : 120
#field of view transaxial        : 432.4
#field of view axial                : 956.8
voxels number transaxial: 192
voxels number axial: 89
field of view transaxial: 600 # given in mm
field of view axial: 250.4 # given in mm

description        : PET system extracted from GATE macro: GE_geometry_layer.mac

scanner radius : 311.8
number of rsectors              : 28
number of crystals transaxial    : 4
number of crystals axial            : 9

crystals size depth                : 25
crystals size transaxial          : 3.95
crystals size axial                 : 5.3


# OPTIONAL FIELDS
rsectors first angle              : 0
number of rsectors axial            : 1
rsector gap transaxial                : 0
rsector gap axial                        : 0
number of modules transaxial    : 1
number of modules axial            : 5
module gap transaxial                : 0
module gap axial                        : 0
#number of submodules transaxial    : 1
#number of submodules axial            : 4
number of submodules transaxial    : 4
number of submodules axial            : 1
submodule gap transaxial                : 0
submodule gap axial                        : 0
crystal gap transaxial                : 0.0333
crystal gap axial                        : 0
mean depth of interaction       :  -1
rotation direction       : CCW 

