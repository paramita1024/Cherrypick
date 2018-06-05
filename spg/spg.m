#     =================================================================
#     File: spg.m
#     =================================================================
#
#     =================================================================
#     Module: Spectral Projected Gradient Method
#     =================================================================
#
#     Last update of any of the component of this module:
#
#     January 15, 2008.
#
#     Users are encouraged to download periodically updated versions of
#     this code at the TANGO Project web page:
#
#     www.ime.usp.br/~egbirgin/tango/
#
#     =================================================================
#     =================================================================

function reperr (inform)

#    Reports errors that occours during sevalf, sevalg and sevalg executions.
# 
#    Name   (i/o)  Description:
#
#    inform (i)    Information Parameter

  if (inform == -90)
   printf("\n\n*** There was an error in the user supplied EVALF function\n");
  elseif (inform == -91)
   printf("\n\n*** There was an error in the user supplied EVALG function\n");
  elseif (inform == -92)
   printf("\n\n*** There was an error in the user supplied PROJ function\n");
  endif 
endfunction

function [f, inform] = sevalf (n, x, inform)

#   Function sevalf evaluates the function f at point x.
#
#    Name  (i/o)  Description:
#
#    n     (i)    size of problem.
#
#    x     (i)    point to be evaluate.
#
#    f     (o)    Value of f(x).
#
#  inform  (i/o)   Information Parameter.
#                  0 = No problem occours.
#                  -90 = Some problem occours on function evaluation. 

   [f, flag] = evalf(n,x);

   if (flag != 0) 
    inform = -90;
    reperr (inform);
   endif
endfunction

function [g, inform] = sevalg (n, x, inform)

#   Function sevalg evaluates the gradient vetor of f at point x.
#
#    Name  (i/o)  Description:
#
#    n     (i)    size of problem.
#
#    x     (i)    point to be evaluate.
#
#    g     (o)    Value of f(x).
#
#  inform  (i/o)   Information Parameter.
#                  0 = No problem occours.
#                  -91 = Some problem occours on function evaluation. 

   [g, flag] = evalg(n,x);

   if (flag != 0)
    inform = -91;
    reperr (inform);
   endif
endfunction

function [x, inform] = sproj (n, x, inform)

#    Function sproj evaluates the projection of point x onto feasible set.
#
#    Name   (i/o)  Description:
#
#    n      (i)    size of problem.
#
#    x      (i/o)    point to be evaluate.
#
#    inform (i/o)  Information Parameter.
#                  0 = No problem occours.
#                 -92 = Some problem occours on function evaluation. 

   [x, flag] = proj(n,x);

   if (flag != 0)
    inform = -92;
    reperr (infom);
   endif
endfunction

function[fcnt, fnew, xnew, lsinfo, inform] = linesearch (n, x, f, g, d, M, lastfv, maxfc,fcnt, inform)

#     Function linesearch implements a nonmonotone line search with
#     safeguarded quadratic interpolation.
#
#    This version 17 JAN 2000 by E.G.Birgin, J.M.Martinez and M.Raydan.
#    Reformatted 25 FEB 2008 by Tiago Montanher.
#    Final revision 30 APR 2001 by E.G.Birgin, J.M.Martinez and M.Raydan.
#
#    Name  (i/o)  Description:
#
#    n      (i)    size of the problem.
#
#    x      (i)    initial guess.
#
#    f      (i)    function value at the actual point.
#
#    d      (i)    search direction.
#
#    g      (i)    gradient function evaluated at initial guess.
#
#    M      (i)    number of previous function values to be considered 
#                  in the nonmonotone line search.
#
#    lastfv (i)    last m function values.
#
#    maxfc  (i)    maximum number of function evaluations.
#
#    fcnt   (i/o)  actual number of fucntion evaluations.
#
#    fnew   (o)    function value at the new point.
#
#    xnew   (o)    new point.
#     
#    lsinfo:
#
#     0: Armijo-like criterion satisfied
#     2: Maximum number of functional evaluations reached
#
#    inform (i/o)    Information parameter:
#	            0= no problem occours during function evaluation,
#	           -90=  some problem occours during function evaluation,  

   sigma_max = 0.1;
   sigma_min = 0.9;
   gamma = 10^-4;

   fmax = max(lastfv);
   gtd = g' * d;

   alpha = 1.0;
   xnew = x + alpha * d;

   [fnew, inform] = sevalf(n, xnew, inform);
   fcnt = fcnt + 1;
   if (inform != 0) return; endif
  
   while (fnew > fmax + gamma * alpha * gtd & fcnt < maxfc)
     if (alpha <= sigma_min) alpha = 0.5 * alpha;

     else
       a_temp = -0.5 * (alpha^2) * gtd / (fnew - f - alpha * gtd);

       if (a_temp < sigma_min | a_temp > sigma_max * alpha) 
        a_temp = 0.5 * alpha; endif

       alpha = a_temp;
     endif

     xnew = x + alpha * d;
     [fnew, inform] = sevalf(n, xnew, inform);
     fcnt = fcnt + 1;
     if (inform != 0) return; endif
   endwhile 

   if (fnew <= fmax + gamma * alpha * gtd) lsinfo = 0;
   else lsinfo = 2; endif
endfunction

function [x, f, gpsupn, iter, fcnt, spginfo, inform] = spg (n, x, epsopt, maxit, maxfc, iprint)

#     Subroutine SPG implements the Spectral Projected Gradient Method 
#     (Version 2: "Feasible continuous projected path") to find a 
#     local minimizers of a given function with convex constraints, 
#     described in
#
#     E.G. Birgin, J.M. Martinez and M. Raydan, "Nonmonotone spectral
#     projected gradient methods for convex sets", SIAM Journal on
#     Optimization 10, pp. 1196-1211, 2000.
#
#     The user must supply the external subroutines evalf, evalg and 
#     proj to evaluate the objective function and its gradient and to 
#     project an arbitrary point onto the feasible region.
#
#     This version 20 DEC 2007 by E.G.Birgin, J.M.Martinez and M.Raydan.
#     Reformatted 25 FEB 2008 by Tiago Montanher.
#
#     Other parameters (i means input, o means output):
#
#     n       (i)   number of variables
#     x       (i/o) initial guess on input, solution on output
#     epsopt  (i)   tolerance for the convergence criterion
#     maxit   (i)   maximum number of iterations
#     maxfc   (i)   maximum number of functional evaluations
#     iprint  (i)   controls output level (0 = no print)
#     f       (o)   functional value at the solution
#     gpsupn  (o)   sup-norm of the projected gradient at the solution
#     iter    (o)   number of iterations
#     fcnt    (o)   number of functional evaluations
#     spginfo (o)   indicates the reason for stopping
#     inform  (o)   indicates an error in an user supplied subroutine
#
#     spginfo:
#
#     0: Small continuous-projected-gradient norm
#     1: Maximum number of iterations reached
#     2: Maximum number of functional evaluations reached
#
#     spginfo remains unset if inform is not equal to zero on output
#
#     inform:
#
#       0: ok
#     -90: error in the user supplied evalf subroutine
#     -91: error in the user supplied evalg subroutine
#     -92: error in the user supplied proj  subroutine

   if (iprint != 0) 
    printf("==============================================================================\n");
    printf(" This is the SPECTRAL PROJECTED GRADIENT (SPG) for  convex-constrained\n");
    printf(" optimization. If you use this code, please, cite:\n\n");
    printf(" E. G. Birgin, J. M. Martinez and M. Raydan, Nonmonotone spectral projected\n");
    printf(" gradient methods on convex sets, SIAM Journal on Optimization 10, pp.\n");
    printf(" 1196-1211, 2000, and\n\n");
    printf(" E. G. Birgin, J. M. Martinez and M. Raydan, Algorithm 813: SPG - software\n");	
    printf(" for convex-constrained optimization, ACM Transactions on Mathematical\n");
    printf(" Software 27, pp. 340-349, 2001.\n");
    printf(" ==============================================================================\n");
    printf("\n");
    printf(" Entry to SPG.\n");
    printf(" Number of Variables:\t%d\n", n);  
   endif

   inform = 0;
   iter = 0;
   fcnt = 0;

   lambda_min = 10^-30;
   lambda_max = 10^30;
   M = 10;

   lastfv(1:M) = -Inf;
   
   [x, inform] = sproj (n, x, inform);
   if (inform != 0) return; endif

   [f, inform] = sevalf (n,x, inform);
   if (inform != 0) return; endif
   fcnt = fcnt + 1;

   [g, inform] = sevalg (n, x, inform);
   if (inform != 0) return; endif
 
   lastfv(M) = f;   

   [gp, inform] = sproj (n, x - g, inform);
   if (inform != 0) return; endif

   gp = gp - x;

   gpsupn = norm(gp, p = Inf);
                             
   if (gpsupn != 0) lambda = min (lambda_max, max (lambda_min, 1.0 / gpsupn));
   else lambda = 0.0; endif

   while (gpsupn > epsopt & iter < maxit & fcnt < maxfc)  

     if(iprint != 0) 
      if (mod(iter, 10) == 0) 
       printf("\n ITER\t F\t GPSUPNORM\n");
      endif
      printf(" %d\t %e\t %e\n", iter, f, gpsupn); 
     endif

     tabline = fopen ("spg-tabline.out", "w");
     fprintf(tabline, " %d  %d  %d  %e  %e Abnormal termination. Probably killed by CPU time  limit.\n", n, iter, fcnt, f, gpsupn );
     fclose(tabline);

     iter = iter + 1;

     [d, inform] = sproj (n, x - lambda * g, inform);
     if (inform != 0) return; endif

     d = d - x;

     [fcnt, fnew, xnew, inform, lsinfo] = linesearch (n, x, f, g, d, M, lastfv, maxfc, fcnt, inform);

     if (inform != 0) return; endif 

     if (lsinfo == 2) 
      spginfo = 2; 
      if (iprint != 0) 
       printf("Flag of SPG: Maximum of functional evaluations reached.\n");
      endif
      return;
     endif

     f = fnew;

     if(mod(iter, M) == 0) lastfv(M) = f;
     else lastfv(mod(iter, M)) = f; endif

     [gnew, inform] = sevalg (n, xnew, inform);
     if (inform != 0) return; endif

     s = xnew - x;
     y = gnew - g;
     sts = s' * s;
     sty = s' * y;

     x = xnew;
     g = gnew;    

     [gp, inform] = sproj (n, x - g, inform);
     if (inform != 0) return; endif

     gp = gp - x; 
     gpsupn = norm(gp, p = Inf);

     if (sty <= 0) lambda = lambda_max;
     else lambda = min (lambda_max, max(lambda_min, sts / sty)); endif
   endwhile

   if(iprint != 0) 
    if (mod(iter, 10) == 0) 
     printf("\n ITER\t F\t GPSUPNORM\n");
    endif
    printf(" %d\t %e\t %e\n", iter, f, gpsupn); 
   endif

   if (iprint != 0) 
    printf("\n");
    printf ("Number of iterations               : %d\n", iter);
    printf ("Number of functional evaluations   : %d\n", fcnt);
    printf ("Objective function value           : %e\n", f);
    printf ("Sup-norm of the projected gradient : %e\n", gpsupn);
   endif

   if (gpsupn <= epsopt) 
    spginfo = 0; 
    if (iprint != 0)
     printf("Flag of SPG: Solution was found.\n"); 
    endif
    return;
   endif

   if (iter >= maxit) 
    spginfo = 1; 
    if (iprint != 0) 
     printf("Flag of SPG: Maximum of iterations reached.\n"); 
    endif
    return;
   endif

   if (fcnt >= maxfc)
    spginfo = 2; 
    if (iprint != 0)
     printf("Flag of SPG: Maximum of functional evaluations reached.\n"); 
    endif
    return;
   endif   
endfunction
