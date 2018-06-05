#include <oct.h>
#include "eval.h"

DEFUN_DLD (evalg,arguments,nargout,"Evaluate Gradient Function\n") {
  int i, n, flag;
  double f, *x, *g;

  NDArray x_array, g_array;

  octave_value_list return_value;

  n = arguments(0).int_value();

  x = (double *) malloc(n * sizeof(double));
  g = (double *) malloc(n * sizeof(double));

  x_array = arguments(1).array_value();

  for(i = 0; i < n; i++) x[i] = x_array.elem(i);  

  evalg(n, x, g, &flag);

  g_array.resize_no_fill(n);
  for (i = 0; i < n; i++) g_array(i) = g[i];

  return_value(0) = g_array;
  return_value(1) = flag;
  return return_value;
}
