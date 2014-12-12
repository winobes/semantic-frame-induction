TEX= latex -shell-escape

NAME=slides
SRC= $(NAME).tex
PDF= $(NAME).pdf

$(PDF): $(SRC) refs.bib
	$(TEX) -output-format=pdf $(SRC)
	bibtex $(NAME).aux
	$(TEX) -output-format=pdf $(SRC)
	$(TEX) -output-format=pdf $(SRC)

.PHONY: clean
clean:
	rm $(NAME).aux $(NAME).log $(NAME).dvi $(NAME).out $(NAME).nav $(NAME)-blx.bib $(NAME).bbl $(NAME).bcf $(NAME).blg $(NAME).pdf $(NAME).run.xml $(NAME).snm $(NAME).toc





