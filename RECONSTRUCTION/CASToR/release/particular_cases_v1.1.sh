#!/bin/bash

version="1.1"

# The analytical projection tool will be released later
rm castor_v${version}/castor-proj.cc

# The deformation part of the code is not yet validated, so we remove the doc
rm castor_v${version}/docs/CASToR_HowTo__image_deformation.pdf

# The multi-bed merging tool will be released later
rm castor_v${version}/toolkits/castor-mergeBedPositions.cc

