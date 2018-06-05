/* =================================================================
   File: cwrapper.h
   =================================================================

   =================================================================
   Module: Spectral Projected Gradient. CUTEr wrapper.
   =================================================================

   Last update of any of the component of this module:

   March 14, 2008.

   Users are encouraged to download periodically updated versions of
   this code at the TANGO Project web page:

   www.ime.usp.br/~egbirgin/tango/

   =================================================================
   ================================================================= */

#include "cfortran.h"

FCALLSCSUB12(spg,SPG,spg,INT,DOUBLEV,DOUBLE,INT,INT,INT,\
	     PDOUBLE,PDOUBLE,PINT,PINT,PINT,PINT)

PROTOCCALLSFSUB4(EVALF,evalf,INT,DOUBLEV,PDOUBLE,PINT)
PROTOCCALLSFSUB4(EVALG,evalg,INT,DOUBLEV,DOUBLEV,PINT)
PROTOCCALLSFSUB3(PROJ,proj,INT,DOUBLEV,PINT)

#define Evalf(n,x,f,flag) \
CCALLSFSUB4(EVALF,evalf,INT,DOUBLEV,PDOUBLE,PINT,n,x,f,flag)

#define Evalg(n,x,g,flag) \
CCALLSFSUB4(EVALG,evalg,INT,DOUBLEV,DOUBLEV,PINT,n,x,g,flag)

#define Proj(n,x,flag) \
CCALLSFSUB3(PROJ,proj,INT,DOUBLEV,PINT,n,x,flag)

