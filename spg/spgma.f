C     =================================================================
C     File: spgma.f
C     =================================================================

C     =================================================================
C     Module: Spectral Projected Gradient Method. Main program.
C     =================================================================

C     Last update of any of the component of this module:

C     March 14, 2008.

C     Users are encouraged to download periodically updated versions of
C     this code at the TANGO Project web page:
C
C     www.ime.usp.br/~egbirgin/tango/

C     *****************************************************************
C     *****************************************************************

      program spgma

      implicit none

C     PARAMETERS
      integer nmax
      parameter ( nmax = 100000 )

C     LOCAL SCALARS
      integer fcnt,i,inform,iter,iprint,maxfc,maxit,n,spginfo
      double precision f,gpsupn,epsopt
      real time

C     LOCAL ARRAYS
      double precision x(nmax)
      real dum(2)

C     DATA STATEMENTS
      data dum/0.0,0.0/

C     Set problem data

      call inip(n,x)

C     Open output file

      open(10,file='spg.out')

C     Set solver parameters

      iprint = 1
      maxit  = 50000
      maxfc  = 10 * maxit
      epsopt = 1.0d-06

C     Call SPG

      time = dtime(dum)

      call spg(n,x,epsopt,maxit,maxfc,iprint,f,gpsupn,iter,fcnt,spginfo,
     +inform)

      time = dtime(dum)
      time = dum(1)

C     Write statistics

      write(* ,fmt=1010) time
      write(10,fmt=1010) time

C     Close output file

      close(10)

C     Save solution

      open(20,file='solution.txt')

      write(20,2000)
      do i = 1,n
          write(20,fmt=2010) i,x(i)
      end do

      close(20)

C     Save table line

      open(30,file='spg-tabline.out')
      write(30,fmt=3000) n,iter,fcnt,f,gpsupn,spginfo,time
      close(30)

      stop

C     Non-executable statements

 1010 format(/,1X,'Total CPU time in seconds: ',F8.2)
 2000 format(/,'FINAL POINT:',//,2X,'INDEX',16X,'X(INDEX)')
 2010 format(  I7,1P,D24.16)
 3000 format(  1X,I7,1X,I7,1X,I7,1X,1P,D16.8,1X,1P,D7.1,1X,I7,0P,F8.2)

      end

