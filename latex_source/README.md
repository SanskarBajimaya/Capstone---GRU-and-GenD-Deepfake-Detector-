# IEEE LaTeX master source (review package)

**IMPORTANT:** The guideline requires a LaTeX-built two-column IEEE conference PDF, BibTeX, and the actual `latex_source/` GitHub folder screenshot in Appendix A. Word-to-PDF export alone does not meet that requirement.

Files: `main.tex`, `references.bib`, `figures/` and reproducible matplotlib script in the package-level `scripts/` directory. Figures 1-10 are recreated with matplotlib from the original team presentation and repository result CSVs. No restricted FaceForensics++ videos are included.

Compile with `pdflatex main.tex`, `bibtex main`, `pdflatex main.tex`, and `pdflatex main.tex`. Check exact page lengths in your TeX environment. The sample Appendix A graphic is clearly marked as a placeholder, **not a real GitHub screenshot**. Replace the file `figures/fig11_APPENDIX_PLACEHOLDER.png` and its caption after uploading the complete latex_source folder to the public GitHub repository.

The root `scripts/generate_report_figures.py` creates all graphics from package-level `data/*.csv` and does not train any models. Upload the root `scripts/` directory as the assignment requires, together with `latex_source/` and only the permitted derived data.

[NEEDS INPUT] Verify exact local OS build, required additional recent related-work papers, and finalize Appendix A screenshot. Avoid claiming the source has already been committed to GitHub.
