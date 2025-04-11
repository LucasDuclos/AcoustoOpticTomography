#!/bin/bash

########################################################################################################
########################################################################################################
##  U S A G E
########################################################################################################
########################################################################################################
if [ $# != 1 ]
then
  echo ""
  echo "Usage: make_release.sh  do_it"
  echo ""
  echo "This script is used to prepare a clean release. All parameters should be set within the script."
  echo "The only command line option can be 'do_it' to specify that we really want to do it."
  echo ""
  exit 0
fi

# Execute the script only if the provided option is literaly 'do_it'
do_it=$1
if [ "${do_it}" != "do_it" ]
then
  echo "If you want the script to go on, then the option should be 'do_it'."
  exit 0
fi

########################################################################################################
########################################################################################################
##  P A R A M E T E R S   F O R   T H E   R E L E A S E
########################################################################################################
########################################################################################################

# Current contributors
current_contributors="Thibaut MERLIN, Simon STUTE, Didier BENOIT, Claude COMTAT, Marina FILIPOVIC, Mael MILLARDET"
# Past contributors
past_contributors="Valentin VIELZEUF"

# Release version
release_version="2.0.2"

# Project start date (first public release)
start_date="2017"
# Project current date
current_date="2018"

########################################################################################################
########################################################################################################
##  P R E P A R E   F O L D E R S
########################################################################################################
########################################################################################################

#######################################################
# Main folder name
release_folder="castor_v${release_version}"
# Check if it already exists, then throw an error
if [ -d "${release_folder}" ]
then
  echo "***** Release folder '${release_folder}' already exists! Abort."
  exit 1
fi
# Otherwise, create it
mkdir ${release_folder}
if [ $? != 0 ]
then
  echo "***** An error occurred while creating the main release folder '${release_folder}'! Abort."
  exit 1
fi

#######################################################
# Create sub-folders
mkdir ${release_folder}/include ${release_folder}/src ${release_folder}/docs ${release_folder}/toolkits
if [ $? != 0 ]
then
  echo "***** An error occurred while creating sub-folders 'include/' 'src/' and 'docs/'! Abort."
  exit 1
fi

#######################################################
# Create source thematic sub-folders
cd ../include/
for d in *
do
  # Do it in the include directory
  mkdir ../release/${release_folder}/include/${d}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while creating thematic include sub-folders '${d}'! Abort."
    exit 1
  fi
  # Do it in the src directory
  mkdir ../release/${release_folder}/src/${d}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while creating thematic src sub-folders '${d}'! Abort."
    exit 1
  fi
done
cd ../release/

#######################################################
# Copy the config sub-folder entirely
cp -r ../config ${release_folder}/
if [ $? != 0 ]
then
  echo "***** An error occurred while copying the 'config/' folder! Abort."
  exit 1
fi

#######################################################
# Copy makefiles
cp ../CMakeLists.txt ../Makefile ${release_folder}/
if [ $? != 0 ]
then
  echo "***** An error occurred while copying makefiles! Abort."
  exit 1
fi

# Copy cmake modules directory
cp -r ../cmake-modules ${release_folder}/
if [ $? != 0 ]
then
  echo "***** An error occurred while copying makefiles! Abort."
  exit 1
fi

# Copy cmake config file
cp  ../include/management/oCASToRConfig.hh.in ${release_folder}/include/management
if [ $? != 0 ]
then
  echo "***** An error occurred while copying makefiles! Abort."
  exit 1
fi


########################################################################################################
########################################################################################################
##  M A K E   D O C U M E N T A T I O N
########################################################################################################
########################################################################################################

#######################################################
# Go into the original docs/tex_files directory
cd ../docs/tex_files/
if [ $? != 0 ]
then
  echo "***** The 'docs/tex_files/' folder of the git repository is missing! Abort."
  exit 1
fi
# Chmod 755 the make_pdf.sh script
if [ ! -e make_pdf.sh ]
then
  echo "***** The 'make_pdf.sh' script does not exist! Abort."
  exit 1
fi
chmod 755 make_pdf.sh
if [ $? != 0 ]
then
  echo "***** Cannot make the 'make_pdf.sh' script executable! Abort."
  exit 1
fi
# Compile all tex files using the make_pdf.sh script
./make_pdf.sh
if [ $? != 0 ]
then
  echo "***** Some errors occurred while compiling all TEX documents! Abort."
  exit 1
fi
# Copy all created pdf files
cd ../
cp CASToR_*.pdf ../release/${release_folder}/docs/
if [ $? != 0 ]
then
  echo "***** An error occurred while copying all PDF documentation files into the documentation release folder! Abort."
  exit 1
fi 
# Go back into the release folder
cd ../release/

########################################################################################################
########################################################################################################
##  B U I L D   T H E   C O P Y R I G H T   T E X T   F I L E
########################################################################################################
########################################################################################################

#######################################################
# Name of the file
copyright="${release_folder}/COPYRIGHT.TXT"
# Create the file
touch ${copyright}
if [ $? != 0 ]
then
  echo "***** An error occurred while creating copyright file '${copyright}'! Abort."
  exit 1
fi

# Write all information into it
if [ "${start_date}" == "${current_date}" ]
then
  echo "Copyright ${start_date} all CASToR contributors listed below:" >> ${copyright}
else
  echo "Copyright ${start_date}-${current_date} all CASToR contributors listed below:" >> ${copyright}
fi
echo "" >> ${copyright}
echo "    --> current contributors: ${current_contributors}" >> ${copyright}
echo "    --> past contributors: ${past_contributors}" >> ${copyright}
echo "" >> ${copyright}
echo "This is CASToR version ${release_version}." >> ${copyright}

#######################################################
# Copy GNU GPL file
cp GNU_GPL.TXT ${release_folder}/
if [ $? != 0 ]
then
  echo "***** An error occurred while copying the GNU GPL file! Abort."
  exit 1
fi

########################################################################################################
########################################################################################################
##  C O P Y   A L L   S O U R C E   F I L E S   A N D   A D D   H E A D E R
########################################################################################################
########################################################################################################

# The GNU copy permission file
copy="gnu_copy_permission.txt"

# To do the copy properly using loops, we move into the root source directory
cd ../

#######################################################
# Copy all include/*/*.hh files while adding the header

# Loop on the include sub-folders
for d in include/*
do
  # Loop on the .hh files
  for f in ${d}/*.hh
  do
    # First add the /* begin-of-comment sign
    echo "/*" > release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding the sign /* into file ${f}! Abort."
      exit 1
    fi
    # Then copy the content of the GNU copy permission file
    cat release/${copy} >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding GNU copy permissions into file ${f}! Abort."
      exit 1
    fi
    # Then add a blank line
    echo "" >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding a blank line into file ${f}! Abort."
      exit 1
    fi
    # Then copy the content of the copyright file
    cat release/${copyright} >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding copyright content into file ${f}! Abort."
      exit 1
    fi
    # Then add the */ end-of-comment file
    echo "*/" >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding the sign */ into file ${f}! Abort."
      exit 1
    fi
    # And finally copy the content of the source file
    cat ${f} >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding the source content into file ${f}! Abort."
      exit 1
    fi
  done
done

#######################################################
# Copy all src/*/*.cc files while adding the header

# Loop on the src sub-folders
for d in src/*
do
  # Loop on the .cc files
  for f in ${d}/*.cc
  do
    # First add the /* begin-of-comment sign
    echo "/*" > release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding the sign /* into file ${f}! Abort."
      exit 1
    fi
    # Then copy the content of the GNU copy permission file
    cat release/${copy} >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding GNU copy permissions into file ${f}! Abort."
      exit 1
    fi
    # Then add a blank line
    echo "" >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding a blank line into file ${f}! Abort."
      exit 1
    fi
    # Then copy the content of the copyright file
    cat release/${copyright} >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding copyright content into file ${f}! Abort."
      exit 1
    fi
    # Then add the */ end-of-comment file
    echo "*/" >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding the sign */ into file ${f}! Abort."
      exit 1
    fi
    # And finally copy the content of the source file
    cat ${f} >> release/${release_folder}/${f}
    if [ $? != 0 ]
    then
      echo "***** An error occurred while adding the source content into file ${f}! Abort."
      exit 1
    fi
  done
done

#######################################################
# Copy all main program files while adding the header

# Loop on the .hh files
for f in *.cc
do
  # First add the /* begin-of-comment sign
  echo "/*" > release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding the sign /* into file ${f}! Abort."
    exit 1
  fi
  # Then copy the content of the GNU copy permission file
  cat release/${copy} >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding GNU copy permissions into file ${f}! Abort."
    exit 1
  fi
  # Then add a blank line
  echo "" >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding a blank line into file ${f}! Abort."
    exit 1
  fi
  # Then copy the content of the copyright file
  cat release/${copyright} >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding copyright content into file ${f}! Abort."
    exit 1
  fi
  # Then add the */ end-of-comment file
  echo "*/" >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding the sign */ into file ${f}! Abort."
    exit 1
  fi
  # And finally copy the content of the source file
  cat ${f} >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding the source content into file ${f}! Abort."
    exit 1
  fi
done

#######################################################
# Copy the toolkits while adding the header

# Loop on the .cc files
for f in toolkits/*.cc
do
  # First add the /* begin-of-comment sign
  echo "/*" > release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding the sign /* into file ${f}! Abort."
    exit 1
  fi
  # Then copy the content of the GNU copy permission file
  cat release/${copy} >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding GNU copy permissions into file ${f}! Abort."
    exit 1
  fi
  # Then add a blank line
  echo "" >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding a blank line into file ${f}! Abort."
    exit 1
  fi
  # Then copy the content of the copyright file
  cat release/${copyright} >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding copyright content into file ${f}! Abort."
    exit 1
  fi
  # Then add the */ end-of-comment file
  echo "*/" >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding the sign */ into file ${f}! Abort."
    exit 1
  fi
  # And finally copy the content of the source file
  cat ${f} >> release/${release_folder}/${f}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while adding the source content into file ${f}! Abort."
    exit 1
  fi
done

# Change back to the release directory
cd release/

# Finally execute any potential particular stuff to do for this release from a separate script
particular_script="particular_cases_v${release_version}.sh"
if [ -f "${particular_script}" ]
then
  ./${particular_script}
  if [ $? != 0 ]
  then
    echo "***** An error occurred while executing particular script '${particular_script}' ! Abort."
    exit 1
  fi
fi

########################################################################################################
########################################################################################################
##  C O M P I L E   S T A T I C   B I N A R I E S   F O R   L I N U X   A N D   W I N D O W S
########################################################################################################
########################################################################################################

# Go to the release directory
cd ${release_folder}/

# Set OMP mode
export CASTOR_OMP=1
# Set the config dir to be in the working folder (for windows only which hard link this path during compilation)
export CASTOR_CONFIG=config
# Set the static flag
export CASTOR_STATIC=1

#####
# Create a directory for static binaries
static_folder="static_bin"
mkdir ${static_folder}
if [ $? != 0 ]
then
  echo "***** An error occurred while creating folder '${static_folder}'! Abort."
  exit 1
fi

#####
# Compile for linux with static links
make -j 32
if [ $? != 0 ]
then
  echo "***** An error occurred while compiling the code for linux static! Abort."
  exit 1
fi
# Move the binaries and suffix them (restrict to castor-recon and datefile and scanner LUT explorers)
cd bin/
for i in castor-recon castor-datafileExplorer castor-scannerLUTExplorer
do
  mv ${i} ../${static_folder}/${i}_unix64
  if [ $? != 0 ]
  then
    echo "***** An error occurred while moving binary '${i}' into static bin folder '${static_folder}'! Abort."
    exit 1
  fi
done
cd ../
# Remove the bin and build folders
rm -rf bin/ build/

#####
# Set the windows flag to 64 bits
export CASTOR_MINGW=64
# Clean the compilation
make clean
if [ $? != 0 ]
then
  echo "***** An error occurred while cleaning the compilation! Abort."
  exit 1
fi
# Compile for windows with static links
make -j 32
if [ $? != 0 ]
then
  echo "***** An error occurred while compiling the code for windows static! Abort."
  exit 1
fi
# Move the binaries and suffix them (restrict to castor-recon and datefile and scanner LUT explorers)
cd bin/
for i in castor-recon castor-datafileExplorer castor-scannerLUTExplorer
do
  mv ${i} ../${static_folder}/${i}_win64
  if [ $? != 0 ]
  then
    echo "***** An error occurred while moving binary '${i}' into static bin folder '${static_folder}'! Abort."
    exit 1
  fi
done
cd ../
# Remove the bin and build folders
rm -rf bin/ build/

#####
# Set the windows flag to 32 bits
export CASTOR_MINGW=32
# Clean the compilation
make clean
if [ $? != 0 ]
then
  echo "***** An error occurred while cleaning the compilation! Abort."
  exit 1
fi
# Compile for windows with static links
make -j 32
if [ $? != 0 ]
then
  echo "***** An error occurred while compiling the code for windows static! Abort."
  exit 1
fi
# Move the binaries and suffix them (restrict to castor-recon and datefile and scanner LUT explorers)
cd bin/
for i in castor-recon castor-datafileExplorer castor-scannerLUTExplorer
do
  mv ${i} ../${static_folder}/${i}_win32
  if [ $? != 0 ]
  then
    echo "***** An error occurred while moving binary '${i}' into static bin folder '${static_folder}'! Abort."
    exit 1
  fi
done
cd ../
# Remove the bin and build folders
rm -rf bin/ build/

# Go back
cd ../

#####
# Copy the README file that explains how to use the binaries
cp static_bin_readme.txt ${release_folder}/${static_folder}/README
if [ $? != 0 ]
then
  echo "***** An error occurred while copying the README file into the static bin folder '${static_folder}'! Abort."
  exit 1
fi

#####
# Copy the whole static_bin directory as a separate one outside of the castor package
cp -r ${release_folder}/${static_folder} ${release_folder}_bin
if [ $? != 0 ]
then
  echo "***** An error occurred while copying the static bin folder '${static_folder}' outside of the CASToR package! Abort."
  exit 1
fi
# Copy the config directory into this same binary folder
cp -r ${release_folder}/config ${release_folder}_bin/
if [ $? != 0 ]
then
  echo "***** An error occurred while copying the config folder outside of the CASToR package! Abort."
  exit 1
fi

