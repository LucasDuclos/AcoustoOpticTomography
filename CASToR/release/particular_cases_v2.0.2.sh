#!/bin/bash

version="2.0.2"

# The analytical projection tool will be released later
rm castor_v${version}/castor-proj.cc

# The deformation part of the code is not yet validated, so we remove the doc
rm castor_v${version}/docs/CASToR__image_deformation.pdf

# The multi-bed merging tool will be released later
rm castor_v${version}/toolkits/castor-mergeBedPositions.cc

# The makeReplicates tool will be released later
rm castor_v${version}/toolkits/castor-makeReplicates.cc

# The sumImages tool will be released later
rm castor_v${version}/toolkits/castor-sumImages.cc

# Dynamic classes will be released later
rm castor_v${version}/src/dynamic/iPatlakModel.cc
rm castor_v${version}/include/dynamic/iPatlakModel.hh
rm castor_v${version}/src/image/iDeformationRigid.cc
rm castor_v${version}/include/image/iDeformationRigid.hh

# The analytical simulator is not ready
rm -r castor_v${version}/include/analytic_simulator castor_v${version}/src/analytic_simulator

