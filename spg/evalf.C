#include <oct.h>
#include "eval.h"

DEFUN_DLD (evalf,arguments,nargout,"Evaluate Function\n") {
  int i, n, flag;
  double f, *x;

  NDArray x_array;

  octave_value_list return_value;

  n = arguments(0).int_value();

  x = (double *) malloc(n * sizeof(double));

  x_array = arguments(1).array_value();

  for(i = 0; i < n; i++) x[i] = x_array.elem(i);  

  evalf(n, x, &f, &flag);

  return_value(0) = f;
  return_value(1) = flag;
  return return_value;
}
