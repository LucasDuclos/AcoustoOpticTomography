#!/bin/bash

# Compile the main documentation
pdflatex CASToR_general_documentation.tex
bibtex CASToR_general_documentation
pdflatex CASToR_general_documentation.tex
pdflatex CASToR_general_documentation.tex
for ext in aux out log toc blg bbl
do
  rm -f CASToR_general_documentation.${ext}
done
mv CASToR_general_documentation.pdf ../

# Compile all other documentations more related to the code
for i in CASToR__*.tex
do
  # Extract the file name without the extension
  filename=${i%.*}
  # Compile a first time
  pdflatex ${filename}.tex
  # Compile a second time for any references
  pdflatex ${filename}.tex
  # Remove temporary files
  rm -f ${filename}.aux ${filename}.log
  # Move the pdf back into the docs/ folder
  mv ${filename}.pdf ../
done

