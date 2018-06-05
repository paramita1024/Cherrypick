#    =================================================================
#    File: spgma.m
#    =================================================================
#
#    =================================================================
#    Module: Spectral Projected Gradient Method
#    =================================================================
#
#    Last update of any of the component of this module:
#
#    January 15, 2008.
#
#    Users are encouraged to download periodically updated versions of
#    this code at the TANGO Project web page:
#
#    www.ime.usp.br/~egbirgin/tango/
#    =================================================================

# SET EXTERNAL FILES

  source("spg.m");
  source("toyprob.m");

# LOAD VARIABLES NEEDED TO CALL THE SOLVER.

  [n, x] = inip ();

  maxit = 50000;	      # Maximum number of iterations

  maxfc = 10 * maxit;         # Maximum number of function evaluations

  iprint = 1;	              # Print Parameter

  epsopt = 1.0E-6;            # Tolerance for the convergence criterion


# CALL THE SOLVER
  [x, f, gpsupn, iter, fcnt, spginfo, inform] = spg (n, x, epsopt, maxit, maxfc, iprint);

# WRITE STATISTICS
  
  solution = fopen ("solution.txt", "w");
  for (i = 1:n) fprintf(solution, "x[%d] = %e\n", i, x(i)); endfor 
  fclose(solution);


  tabline = fopen("spg-tabline.out", "w");
  fprintf (tabline, " %d  %d  %d  %e  %e  %d\n", n, iter, fcnt, f, gpsupn, spginfo);
  fclose(tabline);
