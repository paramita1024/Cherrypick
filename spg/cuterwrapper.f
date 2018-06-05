C     =================================================================
C     File: cuterwrapper.f
C     =================================================================

C     =================================================================
C     Module: Spectral Projected Gradient Method CUTEr wrapper
C     =================================================================

C     Last update of any of the component of this module:

C     December 20, 2007.

C     Users are encouraged to download periodically updated versions of
C     this code at the TANGO Project web page:
C
C     www.ime.usp.br/~egbirgin/tango/

C     *****************************************************************
C     *****************************************************************

      subroutine inip(n,x)

      implicit none

C     SCALAR ARGUMENTS
      integer n

C     ARRAY ARGUMENTS
      double precision x(*)

C     PARAMETERS
      integer input,iout,nmax
      parameter ( nmax  = 100000 )
      parameter ( input =     55 )
      parameter ( iout  =      6 )

C     COMMON ARRAYS
      double precision l(nmax),u(nmax)

C     COMMON BLOCKS
      common /bounds/ l,u
      save   /bounds/

C     EXTERNAL SUBROUTINES
      external usetup

      open(input,file='OUTSDIF.d',form='FORMATTED',status='OLD')

      rewind input

      call usetup(input,iout,n,x,l,u,nmax)

      close(input)

      end

C     *****************************************************************
C     *****************************************************************

      subroutine evalf(n,x,f,flag)

      implicit none

C     SCALAR ARGUMENTS
      integer flag,n
      double precision f

C     ARRAY ARGUMENTS
      double precision x(n)

C     EXTERNAL SUBROUTINES
      external ufn

      flag = 0

      call ufn(n,x,f)

      end

C     *****************************************************************
C     *****************************************************************

      subroutine evalg(n,x,g,flag)

      implicit none

C     SCALAR ARGUMENTS
      integer flag,n

C     ARRAY ARGUMENTS
      double precision g(n),x(n)

C     EXTERNAL SUBROUTINES
      external ugr

      flag = 0

      call ugr(n,x,g)

      end

C     *****************************************************************
C     *****************************************************************

      subroutine proj(n,x,flag)

      implicit none

C     SCALAR ARGUMENTS
      integer flag,n

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
      common /bounds/ l,u
      save   /bounds/

      flag = 0

      do i = 1,n
          x(i) = max( l(i), min( x(i), u(i) ) )
      end do

      end
