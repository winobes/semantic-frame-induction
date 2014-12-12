
EMDerivation.pdf: EMDerivation.tex
	pdflatex -output-fromat=pdf EMDerivation.tex

model0.pdf: model0.tex
	pdflatex -output-fromat=pdf model0.tex

model1GibbsDerivation.pdf: model1GibbsDerivation.tex
	pdflatex -output-fromat=pdf model1GibbsDerivation.tex 


presentation.pdf: presentation.tex refs.bib
	pdflatex -output-format=pdf presentation.tex 
	bibtex presentation.aux
	pdflatex -output-format=pdf presentation.tex 
	pdflatex -output-format=pdf presentation.tex 


