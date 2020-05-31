#! /bin/bash

name=main

pdflatex ${name}.tex
#bibtex tar2020.aux
pdflatex ${name}.tex
pdflatex ${name}.tex

