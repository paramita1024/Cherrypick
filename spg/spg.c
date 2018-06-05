/* =================================================================
   File: spg.c
   =================================================================

   =================================================================
   Module: Spectral Projected Gradient. Method subroutine.
   =================================================================

   Last update of any of the component of this module:

   March 14, 2008.

   Users are encouraged to download periodically updated versions of
   this code at the TANGO Project web page:

   www.ime.usp.br/~egbirgin/tango/

   =================================================================
   ================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "spg.h"

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) > (b) ? (a) : (b))

#define gamma 1.0e-04

#define lmax  1.0e+30
#define lmin  1.0e-30
#define M     100

/***********************************************************************
 **********************************************************************/

void spg(int n,double *x,double epsopt,int maxit,int maxfc,int iprint,
	 double *f,double *gpsupn,int *iter,int *fcnt,int *spginfo,
	 int *inform) {

/* Subroutine SPG implements the Spectral Projected Gradient Method 
   (Version 2: "Feasible continuous projected path") to find a 
   local minimizers of a given function with convex constraints, 
   described in

   E.G. Birgin, J.M. Martinez and M. Raydan, "Nonmonotone spectral
   projected gradient methods for convex sets", SIAM Journal on
   Optimization 10, pp. 1196-1211, 2000.

   The user must supply the external subroutines evalf, evalg and 
   proj to evaluate the objective function and its gradient and to 
   project an arbitrary point onto the feasible region.

   This version 14 MARCH 2008 by E.G.Birgin, J.M.Martinez and M.Raydan.

   Other parameters (i means input, o means output):

   n       (i)   number of variables
   x       (i/o) initial guess on input, solution on output
   epsopt  (i)   tolerance for the convergence criterion
   maxit   (i)   maximum number of iterations
   maxfc   (i)   maximum number of functional evaluations
   iprint  (i)   controls output level (0 = no print)
   f       (o)   functional value at the solution
   gpsupn  (o)   sup-norm of the projected gradient at the solution
   iter    (o)   number of iterations
   fcnt    (o)   number of functional evaluations
   spginfo (o)   indicates the reason for stopping
   inform  (o)   indicates an error in an user supplied subroutine

   spginfo:

   0: Small continuous-projected-gradient norm
   1: Maximum number of iterations reached
   2: Maximum number of functional evaluations reached

   spginfo remains unset if inform is not equal to zero on output

   inform:

     0: ok
   -90: error in the user supplied evalf subroutine
   -91: error in the user supplied evalg subroutine
   -92: error in the user supplied proj  subroutine */

   int i,lsinfo;
   double fbest,fnew,lambda,sts,sty;
   double *d,*g,*gnew,*gp,*lastfv,*s,*xbest,*xnew,*y;
   FILE *tabline;

   char presentation[]=
   "============================================================================\n"
   " This is the SPECTRAL PROJECTED GRADIENT (SPG) for convex-constrained       \n"
   " optimization. If you use this code, please, cite:                          \n\n"
   " E. G. Birgin, J. M. Martinez and M. Raydan, Nonmonotone spectral projected \n"
   " gradient methods on convex sets, SIAM Journal on Optimization 10, pp.      \n"
   " 1196-1211, 2000, and                                                       \n\n"
   " E. G. Birgin, J. M. Martinez and M. Raydan, Algorithm 813: SPG - software  \n"
   " for convex-constrained optimization, ACM Transactions on Mathematical      \n"
   " Software 27, pp. 340-349, 2001.                                            \n"
   "============================================================================\n\n";

/* ==================================================================
   Initialization
   ================================================================== */

/* Print problem information */

   if ( iprint > 0 ) {
      printf("%s",presentation);
      printf(" Entry to SPG.\n");
      printf(" Number of Variables: %d\n",n);  
   }

/* Get memory */

   lastfv = (double *) malloc (M * sizeof(double));
   d      = (double *) malloc (n * sizeof(double));
   g      = (double *) malloc (n * sizeof(double));
   gnew   = (double *) malloc (n * sizeof(double));
   gp     = (double *) malloc (n * sizeof(double)); 
   s      = (double *) malloc (n * sizeof(double));
   xbest  = (double *) malloc (n * sizeof(double));
   xnew   = (double *) malloc (n * sizeof(double));
   y      = (double *) malloc (n * sizeof(double));

/* Set some initial values: */

/* error tracker */
   *inform = 0;

/* for counting number of iterations as well as functional evaluations */
   *iter = 0;
   *fcnt = 0;

/* for the non-monotone line search */
   for ( i=0; i<M; i++ ) 
      lastfv[i] = -1.0e+99;

/* Project initial guess */

   sproj(n,x,inform); 
   if ( *inform  != 0 ) return;

/* Compute function and gradient at the initial point */
   sevalf(n,x,f,inform); (*fcnt)++;
   if ( *inform != 0 ) return;

   sevalg(n,x,g,inform);
   if ( *inform != 0 ) return;

/* Store functional value for the non-monotone line search */
   lastfv[0] = *f;

/* Compute continuous-project-gradient and its sup-norm */

   for ( i=0; i<n; i++ )
      gp[i] = x[i] - g[i];

   sproj(n,gp,inform);
   if ( *inform != 0 ) return;

   *gpsupn = 0.0;
   for ( i=0; i<n; i++ ) {
      gp[i] -= x[i]; 
      *gpsupn = max( *gpsupn, fabs( gp[i] ) );  
   }

/* Initial steplength */
   if ( *gpsupn != 0.0 )
      lambda = min( lmax, max( lmin, 1.0 / *gpsupn ) );
   else 
      lambda = 0.0;

/* Initiate best solution and functional value */
   fbest = *f;

   for ( i=0; i<n; i++ )
      xbest[i] = x[i];

/* Print initial information */

   if ( iprint >  0 ) {
      if ( (*iter) % 10 == 0 ) 
	 printf("\n ITER\t F\t\t GPSUPNORM\n");
      printf(" %d\t %e\t %e\n",*iter,*f,*gpsupn); 
   }
    
   tabline = fopen ("spg-tabline.out", "w");
   fprintf(tabline, "%d %d %d %e %e",n,*iter,*fcnt,*f,*gpsupn);
   fclose(tabline);

/* ==================================================================
   Main loop
   ================================================================== */


   while( *gpsupn > epsopt && *iter < maxit && *fcnt < maxfc ) {

      /* Iteration */
      (*iter)++;

      /* Compute search direction */

      for ( i=0; i<n; i++ )
	 d[i] = x[i] - lambda * g[i];
   
      sproj(n,d,inform);
      if ( *inform != 0 ) return;  

      for ( i=0; i<n; i++ )
	 d[i]-= x[i];

      /* Perform safeguarded quadratic interpolation along the 
	 spectral continuous projected gradient */

      ls(n,x,*f,g,d,lastfv,maxfc,fcnt,&fnew,xnew,&lsinfo,inform); 
      if ( *inform != 0 ) return;

      /* Set new functional value and save it for the non-monotone 
	 line search */

      *f = fnew;
      lastfv[(*iter) % M] = *f;
    
      /* Gradient at the new iterate */

      sevalg(n,xnew,gnew,inform);
      if ( *inform != 0 ) return;

      /* Compute s = xnew - x and y = gnew - g, <s,s>, <s,y>, the
         continuous-projected-gradient and its sup-norm */

      sts = 0.0;
      sty = 0.0;
      for ( i=0; i<n; i++ ) {
	 s[i]  = xnew[i] - x[i];
	 y[i]  = gnew[i] - g[i];
	 sts  += s[i] * s[i];
	 sty  += s[i] * y[i];
	 x[i]  = xnew[i];
	 g[i]  = gnew[i];
	 gp[i] = x[i] - g[i];
      }

      sproj(n,gp,inform);
      if ( *inform != 0 ) return;

      *gpsupn = 0.0;
      for ( i=0; i<n; i++ ) {
	 gp[i] -= x[i]; 
	 *gpsupn = max( *gpsupn, fabs( gp[i] ) );
      } 
  
      /* Spectral steplength */

      if ( sty <= 0 )
	 lambda = lmax;
      else 
	 lambda = max( lmin, min( sts / sty, lmax ) );

      /* Best solution and functional value */

      if ( *f < fbest ) {
	 fbest = *f;

	 for ( i=0; i<n; i++ )
	    xbest[i] = x[i];
      }

      /* Print iteration information */

      if ( iprint >  0 ) {
	 if ( (*iter) % 10 == 0 ) 
	    printf("\n ITER\t F\t\t GPSUPNORM\n");
	 printf(" %d\t %e\t %e\n",*iter,*f,*gpsupn); 
      }
    
      tabline = fopen ("spg-tabline.out", "w");
      fprintf(tabline, "%d %d %d %e %e",n,*iter,*fcnt,*f,*gpsupn);
      fclose(tabline);
   }

/* ==================================================================
   End of main loop
   ================================================================== */

/* Finish returning the best point */

   *f = fbest;

   for ( i=0; i<n; i++ )
      x[i] = xbest[i];

/* Write statistics */

   if ( iprint > 0 ) {
      printf("\n");
      printf (" Number of iterations               : %d\n",*iter);
      printf (" Number of functional evaluations   : %d\n",*fcnt);
      printf (" Objective function value           : %e\n",*f);
      printf (" Sup-norm of the projected gradient : %e\n",*gpsupn);
   }

/* Free memory */

   free(lastfv);
   free(d);
   free(g);
   free(gnew);
   free(gp);
   free(s);
   free(xbest);
   free(xnew);
   free(y);
  
/* Termination flag */

   if ( *gpsupn <= epsopt ) { 
      *spginfo = 0;
      if ( iprint > 0 )
	 printf("\n Flag of SPG: Solution was found.\n");
   }

   else if ( *iter >= maxit ) { 
      *spginfo = 1;
      if ( iprint > 0 )
	 printf("\n Flag of SPG: Maximum of iterations reached.\n");
   }

   else { 
      *spginfo = 2;
      if ( iprint > 0 ) 
	 printf("\n Flag of SPG: Maximum of functional evaluations reached.\n");
   }
}

/***********************************************************************
 **********************************************************************/

void ls(int n,double *x,double f,double *g,double *d,double *lastfv,
	int maxfc,int *fcnt,double *fnew,double *xnew,int *lsinfo,
	int *inform) {

/* Nonmonotone line search with safeguarded quadratic interpolation
 
   lsinfo:
 
   0: Armijo-like criterion satisfied
   2: Maximum number of functional evaluations reached */

   int i;
   double alpha,atmp,fmax,gtd;

   fmax = lastfv[0];
   for ( i=1; i<M; i++ )
      fmax = max (fmax, lastfv[i]);

   gtd = 0.0;
   for ( i=0; i<n; i++ ) 
      gtd+= g[i] * d[i];

   alpha = 1.0;

   for ( i=0; i<n; i++ ) 
      xnew[i] = x[i] + alpha * d[i];

   sevalf(n,xnew,fnew,inform); (*fcnt)++;
   if ( *inform != 0 ) return;

/* Main loop */

   while ( *fnew > fmax + gamma * alpha * gtd && *fcnt < maxfc ) {

      /* Safeguarded quadratic interpolation */

      if ( alpha <= 0.1 )
	 alpha /= 2.0;
     
      else { 
	 atmp = ( - gtd * alpha * alpha ) /
  	         ( 2.0 * ( *fnew - f - alpha * gtd ) );

       if ( atmp < 0.1 || atmp > 0.9 * alpha )
	  atmp = alpha / 2.0;

       alpha = atmp;
      }

      /* New trial */

      for ( i=0; i<n; i++ ) 
	 xnew[i] = x[i] + alpha * d[i];

      sevalf(n,xnew,fnew,inform); (*fcnt)++;
      if ( *inform != 0 ) return;       
   }

/* End of main loop */

/* Termination flag */

   if ( *fnew <= fmax + gamma * alpha * gtd )
      *lsinfo = 0;
   else if ( *fcnt >= maxfc )
      *lsinfo = 2;
}
 
/***********************************************************************
 **********************************************************************/

void sevalf(int n,double *x,double *f,int *inform) { 

   int flag;
   evalf(n,x,f,&flag);

   /* This is true if f if Inf, - Inf or NaN */
   if ( ! ( *f > - 1.0e+99 ) || ! ( *f < 1.0e+99 ) )
      *f = 1.0e+99;

   if ( flag != 0 ) {
      *inform = -90;
      reperr(*inform);
   }
}

/***********************************************************************
 **********************************************************************/

void sevalg(int n,double *x,double *g,int *inform) { 

   int flag;

   evalg(n,x,g,&flag);

   if ( flag != 0 ) {
      *inform = -91;
      reperr(*inform);
   }
}

/***********************************************************************
 **********************************************************************/

void sproj(int n,double *x,int *inform) { 

   int flag;

   proj(n,x,&flag);

   if ( flag != 0 ) {
      *inform = -92;
      reperr(*inform);
   }
}

/***********************************************************************
 **********************************************************************/

void reperr(int inform) {

   char str[80];

   strcpy(str,"*** There was an error in the user supplied ");

   if ( inform == -90 )
      strcat(str,"EVALF");

   else if ( inform == -91 )
      strcat(str,"EVALG");

   else if ( inform == -92 )
      strcat(str,"PROJ");

   strcat(str," subroutine ***");

   printf("\n\n%s\n\n",str);
}  
