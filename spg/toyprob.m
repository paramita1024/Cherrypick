#     SPG Test driver. This master file uses SPG to solve
#     an easy (very easy) bound constrained problem.
#
#     This version 02 FEB 2001 by E.G.Birgin, J.M.Martinez and M.Raydan.
#     Final revision 30 APR 2001 by E.G.Birgin, J.M.Martinez and M.Raydan.

#    Initialize the problem. You MUST change it to solve
#    your own problem.
#    Parameters (i means input, o means output):
#
#     n       (o)   number of variables
#     x       (o)   initial guess on input, solution on output
#

function [n, x] = inip ()
    
  n = 10;  

  for (i = 1:n) x(i) = 60.0; endfor
endfunction

# Evaluate the function at point x. You MUST modify this 
# function to solve your own problem. 
#
#    Name  (i/o)  Description:
#
#     n     (i)   Size of Problem.
#
#     x     (i)   Point to be Evaluated.
#
#     f     (o)   Value of f(x).
#
#   flag    (o)   Flag Parameter
#                 0 if no problem occours during evaluation
#                 nonzero value if any problem occours during evaluation.
#

function [f, flag] = evalf (n, x)

 f = 0;
 for (i = 1:n) f = f + x(i)^2; endfor

 flag = 0;
endfunction

# Evaluate the gradient vector at point x. You MUST modify this 
# function to solve your own problem. 
#
#    Name  (i/o)  Description:
#
#     n     (i)   Size of Problem.
#
#     x     (i)   Point to be Evaluated.
#
#     g     (o)   Gradient Vector of f(x).
#
#   flag    (o)   Flag Parameter
#                 0 if no problem occours during evaluation
#                 nonzero value if any problem occours during evaluation.
#

function [g, flag] = evalg (n, x)

 for (i = 1:n) g(i) = 2 * x(i); endfor

 c = columns (g);
 if (c == n) g = g'; endif

 flag = 0; 
endfunction 

# Evaluate the Projected vector at point x. You MUST modify this 
# function to solve your own problem. 
#
#    Name  (i/o)  Description:
#
#     n     (i)   Size of Problem.
#
#     x     (i/o)   Point to be Evaluated.
#
#   flag    (o)   Flag Parameter
#                 0 if no problem occours during evaluation
#                 nonzero value if any problem occours during evaluation.
#

function [x, flag] = proj (n, x)

# Define Bounds. Change the Bounds to solve your own problem.

  for (i = 1:n) 
    l(i) = -100.0;
    u(i) = 50.0;
  endfor

  for (i = 1:n) x(i) = max(l(i), min(x(i), u(i))); endfor

  c = columns (x);
  if (c == n) x = x'; endif

  flag = 0;
endfunction
