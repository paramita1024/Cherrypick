#include <oct.h>
#include "eval.h"

DEFUN_DLD (proj,arguments,nargout,"Evaluate Projected Point\n") {
  int i, n, inform;
  double f, *x;

  NDArray x_array;

  octave_value_list return_value;

  n = arguments(0).int_value();

  x = (double *) malloc(n * sizeof(double));

  x_array = arguments(1).array_value();

  for(i = 0; i < n; i++) x[i] = x_array.elem(i);  

  proj(n, x, &inform);

  x_array.resize_no_fill(n);
  for (i = 0; i < n; i++) x_array(i) = x[i];

  return_value(0) = x_array;
  return_value(1) = inform;
  return return_value;
}
