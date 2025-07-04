###########################################################################
#                                                                         #
#                          READ ITEMS [0] & [1]                           #
#                          BEFORE YOU GO FURTHER                          #
#                                                                         #
# v1.0 - 01.04.2021: First Release (S.K.Yerli)                            #
# v1.1 - 01.01.2022: Updated for tjaa-2.1 release                         #
# v1.2 - 01.09.2024: Updated for tjaa-3.0 release                         #
#                                                                         #
###########################################################################

[0] Description of files in this folder

    Example article produced from "TJAA LaTeX Template"
    -----------------------------------------------------------
    Example.tex - LaTeX source file (Encoding: UTF-8)
    Example.bib - BibTeX source file (Encoding: UTF-8)
    Example.pdf - PDF file of source produced with "pdflatex" command

    Example-Turkce.tex - LaTeX source file - for Turkish (Encoding: UTF-8)
    Example-Turkce.pdf - PDF file of source produced with "pdflatex" command

    "TJAA LaTeX Template"
    -----------------------------------------------------------
    tjaa.cls    - 'LaTeX Class' file
    tjaa.bst    - 'BibTeX Style' file

[1] Usage of TJAA Template

    Below you will find instructions for different typesetting habits.
    Please choose the one suits best for you and apply.

    PERSONAL USAGE
    ----------------
    If you would like to use the template personally please continue
    reading on [A] or [B].

    SERVER WIDE
    ----------------
    If you are a system admin continue at [C] and follow the instructions
    there and then announce the folder paths to users at the end of this
    section.


###########################################################################
#                                                                         #
#                 DIFFERENT WAYS OF USING tjaa.cls TEMPLATE               #
#                                                                         #
###########################################################################

[A] Quick Start - For those who knows what to do with LaTeX
-----------------------------------------------------------

    1) Create a folder for your manuscript (eg. TJAA_Surname.Name).

    2) Copy files under "TJAA LaTeX Template" and "Example.tex" file.

       If you plan to write more than one manuscript create subfolders
       (eg TJAA_Surname.Name/M01, TJAA_Surname.Name/M02 etc) and copy these
       files into each of the subfolder.

    3) Copy your visual materials (only PNG and PDF extension could be used)
       into this folder.

    4) Rename "Example.tex" file to "Surname.Name-MXX.tex" so that it is not
       mixed up with similar named files.

    5) Edit this file with your typesetting habits and tools.
       Please follow all warnings stated in "Sample Article". This way
       corresponding issue will be produced faster.

       Thank you in advance for your understanding.

    6) Produce the PDF file from the LaTeX source file.
       PLEASE USE ONLY THE COMMAND "pdflatex latex_file" -or- CORRESPONDING
       WINDOWS/MAC FUNCTION.
       Make sure that only "warnings" exists in the file with ".log" extension.
       Make sure that at the end of the file a PDF file has been created.

       Open up the PDF file. Important details that you have to check:
       (a) Are all author names and addresses correct?
       (b) Is the order of author list correct?
       (c) Are all figures and tables included?
       (d) Are all captions numbered in correct order?
       (e) Are all cited work (references) listed at the end?
       (f) Are all citations clickable.
       (g) [special to UAK] Are you within the page limit?

    7) We recommend that you create a sub-folder (eg TJAA-v1) within the
       article folder to upload your article into TJAA reviewing process.
       The versioning part "vxx" at the end of the folder name will help you
       to manage the reviewing process.
       In such a folder keep ONLY the following files:

       The files with the same filenames (eg Soyad.Ad-M01):
          (1) Single LaTeX source file with ".tex" extension.
          (2) Single BibTeX file with ".bib" extension.
              It will be ignored if BibTeX style of referencing is not used.
       The files with different filenames:
          (3) All visual material with either ".png" or ".pdf" extension.
              NO OTHER FILE EXTENSIONS ARE ALLOWED.
              Resolution of each file should be at least 300 dpi or higher.

    For TJAA Article
    ----------------
    8) Register to Dergipark -or- signin to Dergipark:
       https://dergipark.org.tr/

    9) Follow the guidelines in:
       https://dergipark.org.tr/journal/2894/submission/start

    For TJAA - Special Issue (i.e UAK)
    ----------------------------------
    8) Signin to "basvur.uak.info.tr" using registered UAK e-mail address.
       (b) Observe UAK calendar.
       (c) If your presentation is accepted for pyblication by the SOC
           then you have to register/login to DergiPark (see above)
	   and submit there.

       Thanks in advance for your contribution.


[B] LaTeX Information for Beginners or End Users
------------------------------------------------

    1) If you know "nothing" or "little" about LaTeX, please follow
       installations and guidelines in the following link:

       http://math.ucr.edu/~huerta/latexforbeginners.html

    2) Similarly, start with the following links for LaTeX Usage:

       https://en.wikibooks.org/wiki/LaTeX/Basics

       http://www.latex-tutorial.com/tutorials/beginners/

       http://www.cis.upenn.edu/~byorgey/precalc/docs/LaTeX-cheat-sheet.pdf

    3) Please read the following links to learn all about LaTeX editing:

       https://en.wikibooks.org/wiki/LaTeX/Bibliography_Management

       http://merkel.zoneo.net/Latex/natbib.php


    Getting help from those who have been using LaTeX will speed the learning
    curve. After some struggle and attempts, you will be able to focus only
    on writing; this is the main philosophy of LaTeX:

        importance is on WHAT you write rather than HOW you write it

    Therefore, copying "Example Article" file mentioned in section [0] and
    filling the sections according to your article completes the most
    important initial stage in producing article using LaTeX. We recommend to
    get help without delay, when you have trouble editing Figure, Table or
    References.

    At this moment you have to complete section [A] by going through the
    guideline by yourself or find someone to do it.

    Thanks in advance for your contribution and understanding.

[C] Server Installation for TJAA LaTeX Template (admin)
-------------------------------------------------------

    No installation file (.ins) is provided for TJAA template. Therefore
    "manual" installation has to be done.
    Until this installation file is available please follow the instruction
    below for the installation of the template.

    1) DOWNLOAD  tjaa-X.XX.zip file

    2) UNZIP the content into a temporary folder:

       Unix/Linux:
       -----------
       cd /tmp
       unzip tjaa-X.XX.zip
       cd tjaa-X.XX

       Windows:
       --------
       * Right-click on the ZIP file.
       * Choose "Extract Here" option (or equivalent).
       * A folder named "tjaa-X.XX" will be created. Go to this folder.

    3) COPY the files into proper folders:

       If you haven't installed an external LaTeX package you have to create a
       local directory structure. It has to be compliant to TDS standards so
       the LaTeX could access to your files.

       For WINDOWS, you could manage all operations using a GUI program
       provided by the LaTeX distribution. Even though it depends on the
       distribution, you should look for "Add/Import external package".

       One way or another construct the following directory structure:
       ---------------------------------------------------------------
       /usr/local/share/
           .../texmf/tex/latex/tjaa/
           .../texmf/bibtex/bib/tjaa/
           ................/bst/tjaa/
           .../texmf/doc/latex/tjaa/

       The folder "/usr/local/share/texmf" has to be defined as TEXMFLOCAL for
       LaTeX. Due to the differences in LaTeX distributions creating or
       defining TEXMFLOCAL will be different. Please do this using your own
       resources.

       Copy Files:
       -----------
       tjaa.cls        => TEXMFLOCAL/tex/latex/tjaa
       tjaa.bst        => TEXMFLOCAL/bibtex/bst/tjaa
       00_* Example*   => TEXMFLOCAL/doc/latex/tjaa

    4) REPRODUCE LaTeX directory hash (database):

       You have to find corresponding directory hash program which is
       different for different distributions.
       For teTeX, TeXlive it is "texhash".

       For WINDOWS equivalent please read the following link:
       http://www.dedoimedo.com/computers/miktex.html

    5) ANNOUNCE

    6) For the advanced LaTeX users:
       https://en.wikibooks.org/wiki/LaTeX/Installing_Extra_Packages

###########################################################################
#                                                                         #
#          IF YOU SPOT MISTAKES OR HAVE COMMENTS AND SUGGESTIONS          #
#                 PLEASE SEND THEM DIRECTLY TO S.K.YERLI                  #
#                             THANK YOU                                   #
#                                                                         #
###########################################################################
