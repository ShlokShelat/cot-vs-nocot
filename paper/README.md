# Paper

LaTeX source for the submitted paper.

- `main.tex`     -- main paper body (introduction through conclusion)
- `appendix.tex` -- appendices A through F
- `references.bib` -- BibTeX references
- `figures/figure1.tex` -- TikZ source for Figure 1

To compile:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
