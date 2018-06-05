#include <oct.h>
#include <stdio.h>
#include "eval.h"

DEFUN_DLD (inip,,nargout,"Start Point\n") {
  int i, n, nmax;
  double f, *x;

  NDArray x_array;

  octave_value_list return_value;
  
  nmax = 100000;
  x = (double *) malloc(nmax * sizeof(double));

  inip(&n, x);

  x_array.resize_no_fill(n);
  for (i = 0; i < n; i++) x_array(i) = x[i];

  return_value(0) = n;
  return_value(1) = x_array;
  return return_value;
}
