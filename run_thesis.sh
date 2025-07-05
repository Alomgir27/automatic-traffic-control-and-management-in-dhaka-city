#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Compiling Main Thesis ===${NC}"

# Change to the thesis_final directory
cd "$(dirname "$0")/thesis_final" || exit 1

# Compile the thesis multiple times for proper references
echo -e "${YELLOW}First compilation...${NC}"
pdflatex main.tex

echo -e "${YELLOW}Second compilation for references...${NC}"
pdflatex main.tex

echo -e "${YELLOW}Third compilation for final output...${NC}"
pdflatex main.tex

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Main thesis compiled successfully!${NC}"
    
    # Clean up unnecessary files
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz *.lof *.lot
    
    echo -e "${GREEN}✓ Cleanup completed!${NC}"
    echo -e "${GREEN}✓ Main Thesis PDF: thesis_final/main.pdf${NC}"
    
    # Show file size
    if [ -f "main.pdf" ]; then
        SIZE=$(du -h main.pdf | cut -f1)
        echo -e "${GREEN}✓ File size: ${SIZE}${NC}"
        
        # Count pages
        PAGES=$(pdfinfo main.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
        if [ ! -z "$PAGES" ]; then
            echo -e "${GREEN}✓ Total pages: ${PAGES}${NC}"
        fi
    fi
else
    echo -e "${RED}✗ Compilation failed. Check the LaTeX errors above.${NC}"
    exit 1
fi

echo -e "${GREEN}=== Done ===${NC}" 