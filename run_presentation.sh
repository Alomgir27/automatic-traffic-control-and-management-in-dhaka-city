#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Compiling Thesis Presentation ===${NC}"

# Change to the correct directory
cd "$(dirname "$0")" || exit 1

# Compile the presentation
echo -e "${YELLOW}Compiling presentation...${NC}"
pdflatex thesis_presentation.tex

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Presentation compiled successfully!${NC}"
    
    # Clean up unnecessary files
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz
    
    echo -e "${GREEN}✓ Cleanup completed!${NC}"
    echo -e "${GREEN}✓ Presentation PDF: thesis_presentation.pdf${NC}"
else
    echo -e "${RED}✗ Compilation failed. Check the LaTeX errors above.${NC}"
    exit 1
fi

echo -e "${GREEN}=== Done ===${NC}"