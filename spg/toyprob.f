C     =================================================================
C     File: toyprob.f
C     =================================================================

C     =================================================================
C     Module: Spectral Projected Gradient Method. Problem definition.
C     =================================================================

C     Last update of any of the component of this module:

C     March 14, 2008.

C     Users are encouraged to download periodically updated versions of
C     this code at the TANGO Project web page:
C
C     www.ime.usp.br/~egbirgin/tango/

C     *****************************************************************
C     *****************************************************************

      subroutine inip(n,x)

C     SCALAR ARGUMENTS
      integer n

C     ARRAY ARGUMENTS
      double precision x(n)

C     PARAMETERS
      integer nmax
      parameter ( nmax = 100000 )

C     COMMON ARRAYS
      double precision l(nmax),u(nmax)

C     LOCAL SCALARS
      integer i

C     COMMON BLOCKS
      common /bounds/l,u

C     Number of variables
      n = 10

C     Initial point
      do i = 1,n
          x(i) = 60.0d0
      end do

C     Bound constraints (or any other constraints that define a 
C     convex set)
      do i = 1,n
          l(i) = - 100.0d0
          u(i) =    50.0d0
      end do

      end

C     *****************************************************************
C     *****************************************************************

      subroutine evalf(n,x,f,flag)

C     SCALAR ARGUMENTS
      double precision f
      integer n,flag

C     ARRAY ARGUMENTS
      double precision x(n)

C     LOCAL SCALARS
      integer i

      flag = 0

      f = 0.0d0
      do i = 1,n
          f = f + x(i) ** 2
      end do

      end

C     *****************************************************************
C     *****************************************************************

      subroutine evalg(n,x,g,flag)

C     SCALAR ARGUMENTS
      integer n,flag

C     ARRAY ARGUMENTS
      double precision g(n),x(n)

C     LOCAL SCALARS
      integer i

      flag = 0

      do i = 1,n
          g(i) = 2.0d0 * x(i)
      end do

      end

C     *****************************************************************
C     *****************************************************************

      subroutine proj(n,x,flag)

C     SCALAR ARGUMENTS
      integer n,flag

C     ARRAY ARGUMENTS
      double precision x(n)

C     PARAMETERS
      integer nmax
      parameter ( nmax = 100000 )

C     COMMON ARRAYS
      double precision l(nmax),u(nmax)

C     LOCAL SCALARS
      integer i

C     COMMON BLOCKS
      common /bounds/l,u

      flag = 0

      do i = 1,n
          x(i) = max( l(i), min( x(i), u(i) ) )
      end do

      end

