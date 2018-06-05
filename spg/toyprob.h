/* =================================================================
   File: toyprob.h
   =================================================================

   =================================================================
   Module: Spectral Projected Gradient. Problem definition.
   =================================================================

   Last update of any of the component of this module:

   March 14, 2008.

   Users are encouraged to download periodically updated versions of
   this code at the TANGO Project web page:

   www.ime.usp.br/~egbirgin/tango/

   =================================================================
   ================================================================= */

void evalf(int n,double *x,double *f,int *flag);

void evalg(int n,double *x,double *g,int *flag);

void proj(int n,double *x,int *flag);
