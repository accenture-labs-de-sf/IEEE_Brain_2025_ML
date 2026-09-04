# Build the THMS two-column manuscript. Needs pdflatex, bibtex and IEEEtran.cls.
# Overleaf ships IEEEtran, so this folder can also be uploaded there as-is.

PAPER = paper

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib ../figures/fig1_overview.pdf
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

pages: $(PAPER).pdf
	@pdfinfo $(PAPER).pdf | grep Pages

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out

.PHONY: all pages clean
