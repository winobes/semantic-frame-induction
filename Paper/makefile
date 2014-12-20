paper.pdf: paper.tex
	pdflatex -output-fromat=pdf paper.tex
	bibtex paper.aux
	pdflatex -output-format=pdf paper.tex 
	pdflatex -output-format=pdf paper.tex 

EMDerivation.pdf: EMDerivation.tex
	pdflatex -output-fromat=pdf EMDerivation.tex

model0EM.pdf: model0EM.tex model0plate.tex
	pdflatex -output-fromat=pdf model0EM.tex

model1Gibbs.pdf: model1Gibbs.tex model1plate.tex
	pdflatex -output-fromat=pdf model1Gibbs.tex 


presentation.pdf: presentation.tex refs.bib model0plate.tex model1plate.tex
	pdflatex -output-format=pdf presentation.tex 
	bibtex presentation.aux
	pdflatex -output-format=pdf presentation.tex 
	pdflatex -output-format=pdf presentation.tex 


