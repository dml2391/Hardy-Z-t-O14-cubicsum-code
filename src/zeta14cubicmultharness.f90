module PARI
  use ISO_C_BINDING, only : C_LONG, C_DOUBLE, C_PTR
  interface

     subroutine pari_init(parisize, maxprime) bind(C,name='pari_init')
       import C_LONG
       integer(kind=C_LONG), VALUE              :: parisize
       integer(kind=C_LONG), VALUE              :: maxprime
     end subroutine pari_init
     !
     subroutine pari_close() bind(C,name='pari_close')
     end subroutine pari_close
     !
     type(C_PTR) function dbltor( r ) bind(C,name='dbltor')
       import C_DOUBLE, C_PTR
       real(kind=C_DOUBLE), VALUE  :: r
     end function dbltor
     !
     !
     real(kind=C_DOUBLE) function rtodbl( x ) bind(C,name='rtodbl')
       import C_DOUBLE, C_PTR
       type(C_PTR), VALUE :: x
     end function rtodbl
     !
     type(C_PTR) function gsqr( x ) bind(C,name='gsqr')
       import C_PTR
       type(C_PTR), VALUE :: x
     end function gsqr
     !
     type(C_PTR) function gfloor( x ) bind(C,name='gfloor')
       import C_PTR
       type(C_PTR), VALUE :: x
     end function gfloor
     !
     type(C_PTR) function gceil( x ) bind(C,name='gceil')
       import C_PTR
       type(C_PTR), VALUE :: x
     end function gceil
     !
     integer(kind=C_LONG) function sizedigit( x ) bind(C,name='sizedigit')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
     end function sizedigit
     !
     integer function gtos( x ) bind(C,name='gtos')
       import C_PTR
       type(C_PTR), VALUE :: x
     end function gtos
     !
     type(C_PTR) function gfrac( x ) bind(C,name='gfrac')
       import C_PTR
       type(C_PTR), VALUE :: x
     end function gfrac
     !
     type(C_PTR) function gadd( x , y) bind(C,name='gadd')
       import C_PTR
       type(C_PTR), VALUE :: x
       type(C_PTR), VALUE :: y
     end function gadd
     !
     type(C_PTR) function gsub( x , y) bind(C,name='gsub')
       import C_PTR
       type(C_PTR), VALUE :: x
       type(C_PTR), VALUE :: y
     end function gsub     
     !
     type(C_PTR) function gmul( x , y) bind(C,name='gmul')
       import C_PTR
       type(C_PTR), VALUE :: x
       type(C_PTR), VALUE :: y
     end function gmul
     !
     type(C_PTR) function gdiv( x , y) bind(C,name='gdiv')
       import C_PTR
       type(C_PTR), VALUE :: x
       type(C_PTR), VALUE :: y
     end function gdiv
     !
     type(C_PTR) function gpow(x, n, prec) bind(C,name='gpow')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       type(C_PTR), VALUE :: n
       integer(kind=C_LONG), VALUE :: prec
     end function gpow
     !
     type(C_PTR) function gprec( x , d) bind(C,name='gprec')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: d
     end function gprec
     !
     type(C_PTR) function gmod( x , y) bind(C,name='gmod')
       import C_PTR
       type(C_PTR), VALUE :: x
       type(C_PTR), VALUE :: y
     end function gmod
     !
     type(C_PTR) function glog( x , prec) bind(C,name='glog')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function glog
     !
     type(C_PTR) function gatan( x , prec) bind(C,name='gatan')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function gatan
     !
     type(C_PTR) function gerfc( x , prec) bind(C,name='gerfc')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function gerfc
     !
     type(C_PTR) function gexp( x , prec) bind(C,name='gexp')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function gexp
     !
     type(C_PTR) function glngamma( x , prec) bind(C,name='glngamma')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function glngamma
     !
     type(C_PTR) function gsqrt( x , prec) bind(C,name='gsqrt')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function gsqrt
     !
     type(C_PTR) function gimag( x ) bind(C,name='gimag')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
     end function gimag
     !
     type(C_PTR) function greal( x ) bind(C,name='greal')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
     end function greal
     !
     type(C_PTR) function stoi(x) bind(C,name='stoi')
       import C_PTR, C_LONG
       integer(kind=C_LONG), VALUE :: x
     end function stoi
     !
     type(C_PTR) function stor(x , prec) bind(C,name='stor')
       import C_PTR, C_LONG
       integer(kind=C_LONG), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function stor
     !
     type(C_PTR) function gpsi( x , prec) bind(C,name='gpsi')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function gpsi
     !
     integer(kind=C_LONG) function itos(x) bind(C,name='itos')
       import C_PTR, C_LONG
       type(C_PTR), VALUE :: x
     end function itos
     !
     type(C_PTR) function Pi2n(x, prec) bind(C,name='Pi2n')
       import C_PTR, C_LONG
       integer(kind=C_LONG), VALUE :: x
       integer(kind=C_LONG), VALUE :: prec
     end function Pi2n
     !
     type(C_PTR) function vecsum(x) bind(C,name='vecsum')
       import C_PTR, C_LONG
       type(C_PTR), DIMENSION(:)  :: x
       !  integer(kind=C_LONG), VALUE :: x
     end function vecsum
     !
      type(C_PTR) function gcos( x , prec) bind(C,name='gcos')
      import C_PTR, C_LONG
      type(C_PTR), VALUE :: x
      integer(kind=C_LONG), VALUE :: prec
    end function gcos
    !
    type(C_PTR) function gerepilecopy(av, z) bind(C,name='gerepilecopy')
      import C_PTR, C_LONG
      integer(kind=C_LONG), VALUE :: av
      type(C_PTR), VALUE :: z
    end function gerepilecopy
    !
    subroutine set_avma( av ) bind(C,name='set_avma')
      import C_LONG
      integer(kind=C_LONG), VALUE  :: av
    end subroutine set_avma
    !
    integer(kind=C_LONG) function get_avma( ) bind(C,name='get_avma')
      import C_LONG
    end function get_avma
  end interface
end module PARI

program mpi_harness
    
   use mpi
   use ISO_C_BINDING, only : C_LONG

   implicit none

   integer, parameter :: ndomains = 4 ! The number of domains
   integer, parameter :: dp = selected_real_kind(33)

   integer :: ierr, nsize, myrank, nprocs, colour, key, errcode
   integer :: comm
   integer :: rank, comm_size
   character(len=20) :: myoutputfile
   character(len=10) :: myformat  

   character(len=45) :: inYT(ndomains)
   real(dp) :: et
   integer(kind=C_LONG) :: numbercalc

   namelist /inputs/ et, numbercalc, inYT 

   call MPI_Init(ierr)
   call MPI_Comm_size(MPI_COMM_WORLD, nsize, ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

   ! Ensure that the number of processes is a multiple of ndomains
   if(mod(nsize, ndomains) /= 0) then
       if(myrank == 0) then
           print *, "The number of processes must be a multiple of ndomains"
       end if
       errcode = 1
       call MPI_Abort(MPI_COMM_WORLD, errcode, ierr)
       stop
   end if

   ! Number of processes per domain
   nprocs = nsize / ndomains
   ! Assignment of the process groups
   colour = myrank / nprocs + 1 ! values: 1, 2, ..., ndomains
   ! Specify ranks in the new communicator
   key = mod(myrank, nprocs)

   ! Create an output file per domain
   if(colour < 10) then
       myformat = "(A13, I1)"
     else ! Assuming only up to 99 domains
       myformat = "(A13, I2)"
     endif
     write(myoutputfile, myformat) "zeta14v6resa-", colour 

   !write(*,*) "Rank: ", myrank, " Colour: ", colour, " Key: ", key, " Output file: ", myoutputfile

   ! Read the input parameters
     open(2,file='inputs4.nml',status='old')
       read(unit = 2, nml=inputs)
     close(2,status='keep')

   !write(*,*) "Rank: ", myrank, " Colour: ", colour, " YT: ", YT, " et: ", et, " numbercalc: ", numbercalc

   ! Create new communicators for subdomains
   call MPI_Comm_split(MPI_COMM_WORLD, colour, key, comm, ierr)

   !---------------------------------------------------------------------
   ! Call the original code
   !---------------------------------------------------------------------
   call compute_hardy_fast(myoutputfile, comm, inYT(colour), et, numbercalc)
   

   call MPI_Comm_free(comm, ierr)
   call MPI_Finalize(ierr)

end program mpi_harness

subroutine compute_hardy_fast(outfile, COMM, myYT, my_et, my_numbercalc)

  use mpi
  use ISO_C_BINDING, only : C_PTR, C_DOUBLE, C_LONG
  use PARI

  implicit none

  integer, parameter :: dp = selected_real_kind(33)
  integer, parameter :: dp1 = selected_int_kind(16)

  ! Subroutine arguments
  character(len=20), intent(in) :: outfile
  integer, intent(in) :: COMM
  character(len=45), intent(in) :: myYT
  real(dp), intent(in) :: my_et
  integer(kind=dp1), intent(in) :: my_numbercalc

  integer :: COMM_SIZE, IERR,RANK
  integer, dimension(MPI_STATUS_SIZE) :: status

  integer       :: N,ip,K,icj(0:50),MIT,totblock,irsp(7,2),i,mc,m1(50),pblocks,mmax
  integer(dp1)  :: NC,jsum,RSN,L(0:50),NC20,JE,JS,JF,KSET,numbercalc,nchalf
  integer(kind=C_LONG) :: apn1,apn2,apn3,apn4,apn5
  integer(kind=C_LONG) :: apdiv    = 100000000
  integer(kind=C_LONG) :: apdiv1   = 10000000
  integer(kind=C_LONG) :: apn      = 30
  integer(kind=C_LONG) :: prec   = 5
  integer(kind=8) :: apcb
  real(dp)      :: x0,th0,t,a,C,CE,fx,es,p,zet(3:13),sp,r,xr(50),thetar(50),tim(0:350)
  real(dp)      :: et,ht,Y,F1,LC,zsum(15),tsum,rsphase,rszsum(15),t1,t2,rsfrac,ain,yphase,GAM,tpp,tpm,transit
  real(dp)      :: phicoeff(0:10,50),fracL(0:50),see(10),xpbloc,fudge,GAMCOF(6)
  real(dp)      :: raecutoff,RN1,rae,RN2,raestartofblock,RN1startofblock
  real(dp)      :: rszsumtot(15),rszsum1(15),rstime,SE,tar(15),aar(15),rsphasear(15),yphasear(15)
  real(dp)      :: z_inp, rz, iz
  real(kind=C_DOUBLE) :: apr1,apr2,apr3,apr4,apr5,apr6
  complex(kind=16)   :: w,coe,coeff(7,0:6),zlarcoeff(-1:6),zsmalcoeff(-1:6)
  complex(kind=16)   :: c1,c2,c3,c4,epi4,errcon,qq,c5,c6,c7
  character(len=45)  :: line,YT
  type(C_PTR)        :: apu,apv,apw1,apw2,apw,apw3,apw4,apw5,apv1,app1,app2,app3,app4,app5,aplog
  type(C_PTR)        :: aps1,aps2,aps3,aps4,aps5
  integer(kind=8)        :: N_INTERNAL, N_MAX, NIS




  COMMON/PARMS/ coeff,zlarcoeff,zsmalcoeff
  COMMON/PARS1/ phicoeff,fracL,xr,L,icj,MIT,mmax
  COMMON/PARS2/ GAM,ZET
  COMMON/PARS4/ P,SP,TPP,TPM,EPI4
  COMMON/PARS5/ m1
  COMMON/PARS6/ tar,aar,ain,yphasear,et,ht,Y,numbercalc,K,ip,mc
  COMMON/PARS7/ see
  COMMON/PARS8/ GAMCOF
  COMMON/PARS9/ YT
  
  !       First initialise a number of standard constants for use in the calculation of ZP
  !       Set p=pi. HIGH PRECISION IS ABSOLUTELY ESSENTIAL FOR THE CALCULATIONS. The potential
  !       for round-off errors is ever present. Hence the necessity for real(33) precision values.
  !       The vagaries of the GNU compiler mean that these standard constants have to be set in weird 
  !       ways to achieve the Zero point 32 decimal accuracy.


  integer       :: iblock,IFACT!,mysum
  integer(dp1)  :: M2,N1,N2,ib,ial,ae,aenums,MT,MTM,MTO
  integer(dp1)  :: aestartofblock,N1startofblock,N4
  real(dp)      :: t6,b,term,X,pcsum(15),e
  real(dp)      :: rmc1,rmc2,rmc3,rmc4,rmc5,rmc6,zsum1(15),zsum2(15),RN4
  real(dp)      :: tmod2pi, tlnXmod2pi

  ! Assign variables
  YT = myYT
  et = my_et
  numbercalc = my_numbercalc

  !CALL MPI_Init(IERR)
  !COMM = MPI_COMM_WORLD
  CALL  MPI_COMM_RANK(COMM, RANK, IERR)
  CALL  MPI_COMM_SIZE(COMM, COMM_SIZE, IERR)

  !call MPI_Op_create(qsum,.true.,mysum,IERR)

  C=1.0
  p=4*ATAN(C)
  mmax=10
  tpp=2*p
  tpm=-tpp


  ! AB_FLAG
  ! z_inp will fail if it's not in [0,10) - just a format issue (that is easy but fiddly to deal with) of z in subroutine real2pari
  ! The same issue applies to 'rsphase'. But I believe both parameters are in this range. 




!  z_inp = -1.1235


!  PRINT*, 'z_inp for pari_erfc = ', z_inp

!  CALL pari_erfc(z_inp,rz,iz)

  ! This calculates erfc(  exp(  z_inp*I*pi/4 )  )
  ! rz is the real part, iz is the imag part.

!  PRINT*, 'REAL(Z) = ', rz
!  PRINT*, 'IMAG(Z) = ', iz


  !       Standard constants sqrt(pi) exp(pi*i/4).

  C=1.0
  sp=sqrt(p)
  epi4=(1.0,1.0)/sqrt(2*C)

  !       More standard constants relating to the zeta function and gamma function for real values.
  !       Double precision is fine for the values of zeta(3-9).

  GAM=0.57721566490153286060D0

  zet(3)=1.20205690315959428540d0
  zet(5)=1.03692775514336992633d0
  zet(7)=1.00834927738192282684d0
  zet(9)=1.00200839282608221442d0
  zet(11)=1.0004941886041194646d0
  zet(13)=1.0001227133475784891d0


  !     ** Coefficients used to calculate the complex error function. If you can make the
  !     intrinsic error function erf work for complex arguments all the below is unnecessary

  C1=((1.0,0.0)*cos(1.0d0)+(0.0,1.0)*sin(1.0d0))*sqrt(2.0/p) 
  C2=((1.0,0.0)*cos(4.0d0)+(0.0,1.0)*sin(4.0d0))*sqrt(2.0/p)        
  C3=((1.0,0.0)*cos(2.25d0)+(0.0,1.0)*sin(2.25d0))*sqrt(2.0/p)
  C4=((1.0,0.0)*COS(6.25d0)+(0.0,1.0)*SIN(6.25d0))*sqrt(2.0/p)
  C5=((1.0,0.0)*cos(1.5625d0)+(0.0,1.0)*sin(1.5625d0))*sqrt(2.0/p)        
  C6=((1.0,0.0)*cos(3.0625d0)+(0.0,1.0)*sin(3.0625d0))*sqrt(2.0/p)
  C7=((1.0,0.0)*COS(5.0625d0)+(0.0,1.0)*SIN(5.0625d0))*sqrt(2.0/p)
  !     error function coefficients start here.

  !     COEFF(1,..) WILL APPLY FOR Z=1

  COEFF(1,0)=(0.969264211944215930381490d0,-0.474147636640994245161680d0)

  !     EXACT VALUE FOR ERF(EXP(-i*PI/4)*1)

  COEFF(1,1)=C1*(1.0,-1.0)
  COEFF(1,2)=C1*(1.0,1.0)
  COEFF(1,3)=-C1*(1.0,-3.0)/3.0d0
  COEFF(1,4)=-C1*(5.0,-1.0)/6.0d0
  COEFF(1,5)=-C1*(11.0,13.0)/30.0d0
  COEFF(1,6)=C1*(9.0d0,-31.0d0)/90.0d0

  !     NOW DO SAME FOR ERF(EXP(-i*PI/4)*2)

  COEFF(2,0)=(1.010311712025489491642626d0,0.273925759463539899021137d0)

  !     EXACT VALUE FOR ERF(EXP(-i*PI/4)*2)

  COEFF(2,1)=C2*(1.0,-1.0)
  COEFF(2,2)=C2*(2.0,2.0)
  COEFF(2,3)=C2*(-7.0,9.0)/3.0d0
  COEFF(2,4)=-C2*(11.0,5.0)/3.0d0
  COEFF(2,5)=C2*(13.0,-109.0)/30.0d0
  COEFF(2,6)=C2*(129.0,-31.0)/45.0d0

  COEFF(3,0)=(1.338389640116239225341021d0,-0.096501782737190712909169d0)

  !     EXACT VALUE FOR ERF(EXP(-i*PI/4)*1.5)

  COEFF(3,1)=C3*(1.0,-1.0)
  COEFF(3,2)=C3*(1.5,1.5)
  COEFF(3,3)=C3*(-7.0,11.0)/6.0d0
  COEFF(3,4)=C3*(-15.0,-3.0)/8.0d0
  COEFF(3,5)=C3*(-13.0,-59.0)/40.0d0
  COEFF(3,6)=C3*(67.0,-53.0)/80.0d0

  COEFF(4,0)=(8.264692402664926098554228d-1,-1.394623184589031908854228d-1)

  !     EXACT VALUE FOR ERF(EXP(-i*PI/4)*2.5)

  COEFF(4,1)=C4*(1.0,-1.0)
  COEFF(4,2)=C4*(2.5,2.5)
  COEFF(4,3)=C4*(-23.0,27.0)/6.0d0
  COEFF(4,4)=C4*(-155.0,-95.0)/24.0d0
  COEFF(4,5)=C4*(313.0,-913.0)/120.0d0
  COEFF(4,6)=C4*(1065.0,65.0)/144.0d0

  COEFF(5,0)=(1.215497305362876593269559d0,-0.344267547866857083642667d0)

  !     EXACT VALUE FOR ERF(EXP(-i*PI/4)*1.25)

  COEFF(5,1)=C5*(1.0,-1.0)
  COEFF(5,2)=C5*(1.25,1.25)
  COEFF(5,3)=C5*(-17.0,33.0)/24.0d0
  COEFF(5,4)=C5*(-245.0,-5.0)/192.0d0
  COEFF(5,5)=C5*(-767.0,-1633.0)/1920.0d0
  COEFF(5,6)=C5*(1665.0,-2335.0)/4608.0d0

  COEFF(6,0)=(1.260051027257371974124166d0,1.664741404414525369657245d-1)

  !     EXACT VALUE FOR ERF(EXP(-i*PI/4)*1.75)

  COEFF(6,1)=C6*(1.0,-1.0)
  COEFF(6,2)=C6*(1.75,1.75)
  COEFF(6,3)=C6*(-41.0,57.0)/24.0d0
  COEFF(6,4)=C6*(-511.0,-175.0)/192.0d0
  COEFF(6,5)=C6*(-143.0,-4561.0)/1920.0d0
  COEFF(6,6)=C6*(37527.0,-17353.0)/23040.0d0

  COEFF(7,0)=(7.873277503318070969692577d-1,1.234468235979886882660918d-1)

  !     EXACT VALUE FOR ERF(EXP(-i*PI/4)*2.25)

  COEFF(7,1)=C7*(1.0,-1.0)
  COEFF(7,2)=C7*(2.25,2.25)
  COEFF(7,3)=C7*(-73.0,89.0)/24.0d0
  COEFF(7,4)=C7*(-315.0,-171.0)/64.0d0
  COEFF(7,5)=C7*(827.0,-3419.0)/640.0d0
  COEFF(7,6)=C7*(12081.0,-879.0)/2560.0d0


  !     NEXT COMPUTE THE COEFFICIENTS RELEVANT TO THE ASYMPTOTIC APPROXIMATION
  !     OF ERF(Z*EXP(-i*PI/4)) WHEN ABS(Z)>>1.

  ZLARCOEFF(-1)=(1.0,1.0)/SQRT(2*p)
  ZLARCOEFF(0)=(1.0,0.0)
  DO I=1,6
     ZLARCOEFF(I)=(2*I-1)*ZLARCOEFF(I-1)/(0.0,2.0)
  ENDDO

  !     NEXT COMPUTE THE COEFFICIENTS RELEVANT TO THE APPROXIMATION
  !     OF ERF(Z*EXP(-i*PI/4)) FOR SMALL ABS(Z)<<1

  ZSMALCOEFF(-1)=(1.0,-1.0)/SQRT(0.5*p)
  ZSMALCOEFF(0)=(1.0,0.0)
  C1=ZSMALCOEFF(0)
  IFACT=1
  DO I=1,6
     IFACT=IFACT*I
     C1=C1*(0.0,1.0)
     ZSMALCOEFF(I)=C1/((2*I+1.0)*IFACT*1.0)
  ENDDO

  !     ** All of the above code starting from ** can be excluded if one is able to access the intrinsic built in 
  !     error function of the GNU fortran compiler to compute erf(Z) with Z=x*exp(+/-I*pi/4) and x>0 to at least double precision.

  !     All ready.
  !     Now this version of the code computes the first part of the Hardy function Z(t) using sequences of Generalised Gaussian sums added
  !     together. This replaces the earlier version which computed the Hardy function Z(t) using a sequences of Quadratic Gaussian sums.
  !     The advantage of using Generalised Gaussian sums or order mc=3,4,5... is that the number of summands can be increased, so that the number
  !     of Gaussian sums to be computed is reduced. This means that the algorithm should work faster as the order mc is increased. In fact
  !
  !     the operational count necessary to compute Z(t) will scale as O((et*t)**(1/(mc+1))*(log(t))**2.33).
  !
  !     So mc=2 the quadratic case goes as (et*t)**(1/3)*(log(t)**2.33), mc=3 the cubic case goes as ((et*t)**(1/4))*(log(t)**2.33), etc.
  !
  !     If you want to use the quadratic case use the specialised program zeta13v2.f90 directly relevant for mc=2.
  !
  !     This code assumes we are going beyond that, so the order of the Gaussian sums mc>2.
  !
  !     Input 6 parameters:
  !     1) t>10^5, the value of t in Z(t) t:real
  !     2) mc integer >2 the order of the Generalized Gaussian sums you will be using.
  !     3)&4) Gaussian sum integers 0<ip<=4 and K=O(1) (a value between 20-50 works fine for K) 
  !     5) et : real, the relative error scale 0<et<<1. The smaller et the more accurate your result but the slower the code. A larger value
  !     gives a less accurate estimate of Z(t) but goes faster. Values around 0.01-0.05 worked to six significant figures for the mc=2 case.
  !     The switchover parameter Y where 1.20> Y > 1.05 : real.
  !     The value of Y determines the point at which the value for MT determining the
  !     Generalised Gauss sum length switches from the formula suitable for when pc is close to unity (pc-1 small)
  !     and when (pc-1)~O(1). One can alter this but it should be greater than 1.05 and no more than
  !     about 1.2. Too small gives rise to numerical errors, too large is computationally inefficient.


!  irsp(1,1)=10000000
!  irsp(2,1)=00000000 
!  irsp(3,1)=00000000
!  irsp(4,1)=56100000
!  irsp(5,1)=00000000


   ! open(2,file='inputs3.nml',status='old')
   ! do i=1,9
   !  read(2,4001) line
   ! enddo
   ! read(2,4002) et
   ! read(2,4003) YT
   ! read(2,4001) line
   ! read(2,4006) numbercalc

   !close(2,status='keep')

  nchalf=2*numbercalc-1 

  open(2,file=outfile,status='old')

  if (numbercalc.gt.8) then
     write(2,*) et
     write(2,*) YT
     write(2,*) numbercalc
     write(2,*) 'Numbercalc greater than 8 - reduce to between 2-8'
     close(2,status='keep')
     stop
  endif   

  write(2,4004) 'et= ',et
  write(2,4005) 't= ',YT
  READ(YT(1:1)  , '(I1)'  )  irsp(7,1)
  READ(YT(3:10)  , '(I8)'  )  irsp(1,1)
  READ(YT(11:18)  , '(I8)'  )  irsp(2,1)
  READ(YT(19:26)  , '(I8)'  )  irsp(3,1)
  READ(YT(27:34)  , '(I8)'  )  irsp(4,1)
  READ(YT(35:42)  , '(I8)'  )  irsp(5,1)
  READ(YT(44:45)  , '(I8)'  )  irsp(6,1)  
   
  CE=10
  C=0.0
  do i=5,1,-1
     C=C+irsp(i,1)/(CE**(8*i-1))
  enddo

  C=C/(CE**1)
  CE=1.0
  C=irsp(7,1)*CE+C
  CE=10.0
  t=(CE**(irsp(6,1)))*C
  write(2,4008) 'fortran t= ',t

!   t value stored but ALSO GO TO TGRABBER. Now for a=sqrt(8*t/pi).

!  irsp(1,1)=18736395  
!  irsp(2,1)=76847091
!  irsp(3,1)=17235962
!  irsp(4,1)=13771225
!  CE=10.0
!  C=0.0
!  do i=4,1,-1
!   C=C+irsp(i,1)/(CE**(8*i-1))
!  enddo
!  C=C+0651/(CE**35)
!  CE=10.0
!  a=(CE**12)*C  

  ! AB_FLAG added the following 4 lines to check the pari routine is
  ! working correctly. See subroutine for 't = ...'


  !      b8 = 6

  !      CALL pari_calc(tlnXmod2pi)


  !      if(RANK .eq. 0) then
  !      print*, 't * lngma(N) mod 2pi = ', tlnXmod2pi
  !      endif





  !      CALL pari_init(10000000_8, 2_8)


  !      apn1=irsp(1,1)
  !      apn2=irsp(2,1)
  !      apn3=irsp(3,1)
  !      apn4=irsp(4,1)
  !      apn5=irsp(5,1)
  !      write(2,*) 'apn1= ',apn1
  !      write(2,*) 'apn2= ',apn2
  !      write(2,*) 'apn3= ',apn3
  !      write(2,*) 'apn4= ',apn4
  !      write(2,*) 'apn5= ',apn5

  !      apu = stor(apn1,prec)  
  !      apv = stor(apdiv1,prec)
  !      apw1 = gdiv(apu,apv)
  !      apr6 = rtodbl(apw1)

  !      write(2,*) 'initial multiplication= ',apr6

  !      apu = stor(apn2,prec)
  !      apv1 = stor(apdiv,prec)
  !      apw2 = gdiv(apu,apv1)
  !      apw2 = gdiv(apw2,apv)
  !      apu = stor(apn3,prec)
  !      apw3=gdiv(apu,apv1)
  !      apw3=gdiv(apw3,apv)
  !      apw3=gdiv(apw3,apv)
  !      apu = stor(apn4,prec)
  !      apw4=gdiv(apu,apv1)
  !      apw4=gdiv(apw4,apv)
  !      apw4=gdiv(apw4,apv)
  !      apw4=gdiv(apw4,apv)
  !      apu = stor(apn5,prec)
  !      apw5=gdiv(apu,apv1)
  !      apw5=gdiv(apw5,apv)
  !      apw5=gdiv(apw5,apv)
  !      apw5=gdiv(apw5,apv)
  !      apw5=gdiv(apw5,apv)
  !      app1 = gmul(apw1, gpow(stoi(10_8), stoi(apn), prec)) 
  !      app2 = gmul(apw2, gpow(stoi(10_8), stoi(apn), prec))
  !      app3 = gmul(apw3, gpow(stoi(10_8), stoi(apn), prec))
  !      app4 = gmul(apw4, gpow(stoi(10_8), stoi(apn), prec))
  !      app5 = gmul(apw5, gpow(stoi(10_8), stoi(apn), prec))


  !      apr6 = rtodbl(app1)
  !      write(2,*) 'first number*10**30= ',apr6
  !      call pari_close
  !      Ashley: this was the source of the orignal problem. I created five global variables app1-5
  !      using PARI and then call pari_close. This had the effect of deleting the values I had created     
  !      for app1-5 when used later.....     

  mc=3
  ip=3
!  et=0.005
!  totblock=115
  KSET=0
  IF (KSET.EQ.0) THEN
     K=MAX0(100,FLOOR(((log(et)/log(0.005))**0.25)*(60*log(t)/log(10.0)-1100.0),dp1))
     totblock=10+MAX0(25,FLOOR(((log(et)/log(0.005))**0.25)*(3.545*log(t))-61.8,dp1))
  ELSE
     K=400
     totblock=188
  ENDIF   
  C=100.0
  Y=111/C
  CALL pari_phases(yphase,rsphase,a)

  do i=1,nchalf
     tar(i)=t+(i-numbercalc)*0.01
     yphasear(i)=yphase+(i-numbercalc)*0.01
     if (yphasear(i).lt.0.0) then
        yphasear(i)=yphasear(i)+tpp
     endif
     rsphasear(i)=rsphase+(i-numbercalc)*(log(0.25*a)-1/(48*t*t))*0.01
     if (rsphasear(i).lt.0.0) then
        rsphasear(i)=rsphasear(i)+tpp
     endif
     aar(i)=a+4*(i-numbercalc)*0.01/(p*a)
  enddo

  !     Next input the values for two phases. First is the Rieman-Siegel phase=theta(t) mod (2pi). 
  !     Here theta(t)=IM(log(Gamma(1/4+i*t/2))-t*log(pi)/2 with the asymptotic formula
  !
  !     theta(t)=(t/2)*log(t/(2*pi))-t/2-pi/8+1/(48*t)+7/(5760*t^3).....
  !
  !     This can be calculated directly, but as theta(t)-2*pi*aint(theta(t)/(2*pi)). However, for
  !     large t aint(theta(t)) is beyond the dynamic range (14-15 digits) of a long integer(dp1).
  !     This will give rise to errors and the calculation is prone to round off. So for a ONE OFF
  !     calculation it is better to compute Riemann-Siegel phase=theta(t) mod (2pi) separately, using either
  !     Matlab or Maple, and input the precise result to 20 decimal places at the start. Successive digits of
  !     the theta(t) mod (2pi) calculation are stored in irsp(1-4,1). NB care must be taken if theta(t) mod (2pi)  
  !     <0.1. The code interprets the first digit say 2 as 2.something. If you want to input 0.7 something start
  !     with a 7... and the code divides the 7. something by 10 at the end since 7.something>(2*pi). If you want
  !     0.0something, you must divide by 100 etc.

! irsp(1,1)=601278    
! irsp(2,1)=347676
! irsp(3,1)=147733       
! irsp(4,1)=871377

  !     Exactly the same idea as above, only irsp(1-4,2) stores yphase=t+pi/8 mod(2pi), the constant phase value
  !     that appears in the ZP(t) portion of the hybrid sum for Z(t).

! irsp(1,2)=268140     
! irsp(2,2)=937892     
! irsp(3,2)=727580    
! irsp(4,2)=946098 

! CE=10
! rsphase=0.0
! do i=4,1,-1
!    rsphase=rsphase+irsp(i,1)/(CE**(6*i-1))
! enddo
! if (rsphase.gt.(2*p)) then
!    rsphase=rsphase/CE
! endif

  !     Divide by 10 to bring rsphase below (2*pi)

! yphase=0.0
! do i=4,1,-1
!    yphase=yphase+irsp(i,2)/(CE**(6*i-1))
! enddo
! if (yphase.gt.(2*p)) then
!    yphase=yphase/CE
! endif

  !     Exactly the same idea as above, only irsp(1-4,2) stores tphase=t mod(2pi), the constant phase value
  !     that appears in the ZP(t) portion of the hybrid sum for Z(t).

!  irsp(1,2)=072582 
!  irsp(2,2)=478172
!  irsp(3,2)=201729
!  irsp(4,2)=036107

!  CE=10.0
!  tphase=0.0
!  do i=4,1,-1
!     tphase=tphase+irsp(i,2)/(CE**(6*i-1))
!  enddo
!  if (tphase.gt.(2*p)) then
!     tphase=tphase/CE
!  endif

  !     Derived constants from the preset values of t, et, mc and Y.
  !     a, 1/a, ht, & aecutoff.
  !     totblock gives an estimate of the number of blocks the calculation will be broken down into.
  !     The actual value is calculated in subroutine ZP as the calculation progresses.

!  a=sqrt(8*t/p)
  ain=1/a
  ht=log(et)/log(t)

  raecutoff=exp((log(et)+mc*log(t))/(mc+1.0d0))/(sp*log(t))
  pblocks=14
  xpbloc=(1-pblocks/log(t))/(mc+1.0)
  fudge=(et/(t**(1/(mc+2.0))))**xpbloc
  F1=1+fudge*(1-0.25/(Y*Y))/(2.0**(3*(mc-2)/(2*(mc+1.0d0))))
  LC=((mc-1)*log(t)/(2*mc+2.0d0)-log(log(t))+log((et**(1/(mc+1.0d0)))/(Y*sqrt(8.0d0))))/log(F1)
  !  totblock=nint(0.6*LC)

  !     write(6,3001) 'Current t value= ', t
  !     write(6,3002) 'Relative error scale= ',et,'Cut off integer for QG sums= ',K
  !     write(6,3003) 'Cut off alpha integer= ',raecutoff,'Estimate of number of blocks needed= ',totblock

  !     Output data to a file. Can be done from the command line of course.

  !      open(2,file='zeta14v3resa',status='old')

  write(2,3001) 'Current t value= ', t
  write(2,3001) 'Current a value= ', a
  write(2,3002) 'Relative error scale= ',et,'Cut off integer for QG sums= ',K
  write(2,3003) 'Cut off alpha integer= ',raecutoff,'Estimate of number of blocks needed= ',totblock
  write(2,3009) 'Riemann-Siegel phase= ',rsphase
  write(2,3009) 'pc phase            = ',yphase,'Mmax= ',mmax
  write(2,*) 'All the above parameters pertain to the central t value - not those above or below'
  write(2,*) '******************************************************************************************'

  !     errcon is a constant needed to complete the evauation of the additive term. It depends upon the 
  !     set value of ip.

  errcon=(1.0,0.0)
  do i=1,ip
     errcon=errcon*(0.0,-1.0)
  enddo
  errcon=errcon*exp(-ip*1.0)*(ip**(ip*1.0))*(0.0,1.0)/(p**(ip+1.5))

  !     Compute partial sum ZP(t)=zsum of the hybrid formula and time taken to do this.

  call cpu_time(t1)
  C=1.0
  t6=(sqrt(p/(8*C))*a)**(1/(3*C))
  call start(a,t6,yphase,transit,M2)
  !      write(6,2001) 'Initial alpha value= ',M2
  write(2,2007) 'Block 0. Initial alpha value= ',M2
  ib=2*floor(t**(1/(mc+2.0d0)),dp1)
  if (mod(ib,2).gt.0.5) then
     ib=ib+1
  endif
  N1=M2+ib
  !      write(6,*) 'Final alpha of Block zero= ',N1
  RN1=real(N1,dp)
!  c=sqrt(8/a)
  do i=1,nchalf
     zsum(i)=transit
  enddo
  do ial=M2,N1,2
     do i=1,nchalf
      call ter(ial,1/aar(i),yphasear(i),tar(i),term)
      zsum(i)=zsum(i)+term
     !       if (ial.lt.(M1+12)) then
     !        write(6,*) ial,zsum,term
      !       endif
     enddo   
  enddo
  do i=1,nchalf
     zsum(i)=zsum(i)*sqrt(8/aar(i))
  enddo   
  call cpu_time(t2)
  tim(0)=t2-t1
  write(2,2003) 'zsum after this block= ',zsum(numbercalc)
  write(2,2012) 'alpha value at end of this block= ',N1,'Cpu time needed for this block= ',tim(0)
  write(2,*) '----------------------------------------------------------------------'

  X=exp(((1/(mc+2.0d0)-ht)/(mc+1.0d0)))


  rmc1=1/(mc+1.0)
  rmc2=(mc-1.0)*rmc1
  rmc3=(2*mc-1)*rmc1/2.0
  rmc4=(3*(mc-3)/2.0)*rmc1
  rmc5=1/(mc-1.0)
  rmc6=((4*mc-5.0)/2.0)*rmc1
  t6=(t**(0.5*rmc2))
  do i=1,nchalf
   zsum1(i)=0.0
   zsum2(i)=0.0
  enddo

!      If a restart is necessary activate code below.       
! zsum=4913.412791572d0
! write(2,2003) 'zsum after this block= ',zsum 

! irsp(1,2)=439482    
! irsp(2,2)=239579     
! irsp(3,2)=166717    
! irsp(4,2)=170000 
! CE=10
! RN1=0.0
! do i=3,1,-1
!    RN1=RN1+irsp(i,2)/(CE**(6*i-18))
! enddo
! write(2,2018) 'Alpha value restarts at= ',RN1
! aenums=7219310

  CALL pari_init(10000000_8, 2_8)  ! change 1

  do iblock=1,totblock          ! totblock AB_FLAG

     if(RANK .eq. 0) then

        write(6,*) 'iblock= ',iblock

     endif

     call cpu_time(t1)
     if (RN1.lt.(Y*a)) then
        aenums=floor((X**iblock)*ib/2.0,dp1)
        e=RN1/a
        MT=floor(((((et/p)**rmc5)*a)**rmc2)*((e-1.0)**rmc3)/(2.0**rmc4),dp1)         
     else
        MTO=MT
        b=a/(2.0*RN1)
        MT=floor((et**rmc1)*t6*(1/b-b)/(sp*(2.0**rmc6)),dp1)
        if (MT.gt.(1.5*MTO)) then
         MT=floor(1.5*MTO,dp1)
        endif
        if (iblock.gt.100) then
           mmax=8
        endif
     endif
     if (mod(MT,2).lt.0.25) then
        MT=MT-1
     endif

     write(2,2002) 'Block number= ',iblock,' Number of Gsums= ',aenums,'Length of Gsums= ',MT   
     MTM=(MT-1)/2
     rae=RN1+2.0+real(MT,dp)   

     raestartofblock=RN1+2.0+real(MT,dp)
     RN1startofblock=RN1

     do i=1,nchalf
        zsum1(i)=0.0
     enddo   


!     print*, 'aenums = ', aenums

     do jsum=(rank+1),aenums,COMM_SIZE   !AB_FLAG 




!        if( MOD(jsum,10000) .eq. 0 ) then

!           print*, 'jsum =  ', jsum

!        endif



        RN4=2.0*(real(MT,dp)+1)*(real(jsum-1,dp))
        rae=raestartofblock  + RN4            ! LINE ONE
        RN1=RN1startofblock  + RN4            ! LINE TWO 




        !        if (iblock.eq.1) then
        !         write(2,2014) 'jsum rn4 and MT=',jsum,rn4,MT
        !         write(2,2015) 'rae and rn1=',rae,rn1
        !       endif

        call alphasum(MTM,rae,pcsum)
        do i=1,nchalf
           zsum1(i)=zsum1(i)+pcsum(i)
        enddo   


     enddo



     !       RN1=RN1startofblock + min(2*(MT+1)*aenums*1.0,raecutoff)
     RN1=RN1startofblock + (2*(real(MT,dp)+1.0)*real(aenums,dp))
     !       if (iblock.eq.4) then
     !        write(2,2016) 'at finish ANENUMS=',aenums
     !        write(2,2017) 'rn1 rn1start and 2*(mt+1)*AENUMS=',rn1,RN1startofblock,(2*(MT*1.0+1.0)*aenums)
     !       endif


   !   do i=1,nchalf
   !    !CALL MPI_Barrier(COMM, IERR)
   !    CALL MPI_ALLREDUCE(zsum1(i),zsum2(i),1,MPI_REAL16,MPI_SUM,COMM,IERR)
   !    zsum(i)=zsum(i)+zsum2(i)
   !   enddo 
     CALL MPI_ALLREDUCE(zsum1,zsum2,nchalf,MPI_REAL16,MPI_SUM,COMM,IERR)
     zsum=zsum+zsum2  
      
     !         if(RANK .eq. 0 .and. jsum .ge. (aenums-COMM_SIZE)) then

     !         write(6,*) 'zsum outside after this block1= ',zsum
     !         write(6,*) 'zsum2 outside after this block1= ',zsum2


     !         endif


     call cpu_time(t2)
     tim(iblock)=t2-t1

     write(2,2003) 'zsum after this block= ',zsum(numbercalc)
     write(2,2008) 'alpha value at end of this block= ',RN1,'Cpu time needed for this block= ',tim(iblock)
     write(2,*) '----------------------------------------------------------------------'
  enddo

  CALL pari_close()  ! change 2
  
  raecutoff=RN1
  tsum=0.0
  do i=0,totblock
     tsum=tsum+tim(i)
  enddo
    do i=1,nchalf
       write(2,3004) 'Final value of ZP=', zsum(i)
    enddo
     
  write(2,3005) 'Total cpu time needed for this calculation= ',tsum  

  call cpu_time(t1)
  do i=1,nchalf
   rszsum(i)=0.0
   rszsum1(i)=0.0
   rszsumtot(i)=0.0
  enddo
  CE=4.0
  C=1.0
  CE=raecutoff*(1-sqrt(1-(a/(raecutoff*C))**2))/CE
  NC=aint(CE,dp1)
  write(2,3006) 'NC cut off integer of RS part of sum= ',NC
  !     CALL pari_init(10000000_8, 2_8)
  !    Ashley: Here I restarted PARI to use the global app1-5 variables created earlier, only the code didn't know what they were
  !    causing the program to crash originally. Deleting the close and restart PARI seemed to fix things. But I inadvertently had it
  !    printing out data below trying to figure this out and it suddenly started printing out loads of data to zeta14v3resa
  !    leading to the error messages I emailed you about.

  CE=1.0

!  print*, 'NC = ', NC

  DO IBLOCK=1,1


  ! AB_FLAG - this routine does N_INTERNAL iterations inside the pari routine
  ! jsum = 1, loops from 1:N_internal
  ! jsum = 2, loop from [ (1:N_internal) + N_internal ]
  ! We round up the amount of iterations to the nearest N_internal
     
     NIS = 0
     N_INTERNAL = 100000
     N_MAX = floor( NC / (N_INTERNAL*1.0) )
!     PRINT*, 'NUMBER OF ITERATION = ', N_MAX

!     N_MAX = 128  ! this is for testing. N_MAX should be around 20,000. Assuming this is run on 1 node (128 procs).

     DO jsum=(RANK+1),N_MAX,COMM_SIZE

        CALL pari_calc(rszsum,jsum,numbercalc,rsphasear,N_INTERNAL,NIS)

 !       rszsum = rszsum + rszsum1

     ENDDO

   !   do i=1,nchalf
   !    rszsum1(i) = 0.0


   !    !CALL MPI_Barrier(COMM,IERR)
   !    CALL MPI_ALLREDUCE(RSZSUM(i),RSZSUM1(i),1,MPI_REAL16,MPI_SUM,COMM,IERR)
   !    rszsumtot(i)=rszsumtot(i)+rszsum1(i)
   !   enddo 
     rszsum1 = 0.0
     CALL MPI_ALLREDUCE(RSZSUM,RSZSUM1,nchalf,MPI_REAL16,MPI_SUM,COMM,IERR)
     rszsumtot=rszsumtot+rszsum1

!  Additional bit of code to do the final bit of calculation written by DML

   !   do i=1,nchalf
   !    rszsum1(i) = 0.0
   !   enddo 
      rszsum1 = 0.0
      NIS=N_MAX*N_INTERNAL+1
!     print*,'final start value = ',NIS
      N_INTERNAL=NC
!     print*, 'final end value = ', N_INTERNAL
      jsum=1
      CALL pari_calc(rszsum1,jsum,numbercalc,rsphasear,N_INTERNAL,NIS)

      do i=1,nchalf
       rszsumtot(i)=rszsumtot(i)+rszsum1(i)
      enddo   
   ENDDO


!   PRINT*, 'rszsumtot = ', rszsumtot


  
  !     CALL pari_close

  !     Ashley: Single call to close PARI when all calculations complete.

  do i=1,nchalf 
   RSZSUM(i)=RSZSUMTOT(i)
   CE=sqrt(tar(i)/(2*p))
   RSN=aint(CE,dp1)
   rsfrac=CE-RSN
   CE=sqrt(1/CE)
   C=cos(2*p*(rsfrac*(rsfrac-1.0)-0.0625))/cos(2*p*rsfrac)
   rszsum(i)=2*rszsum(i)+((-1)**(RSN-1))*CE*C
  enddo
  !     Rszsum is the contribution to Z(t) from main part of RS formula plus O(t^-0.25) correction.

  call cpu_time(t2)
  rstime=t2-t1
  !      write(2,3006) 'NC cut off integer of RS part of sum= ',NC
  do i=1,nchalf
   write(2,3007) 'Value of RS contribution to sum= ', rszsum(i)
  enddo   
  write(2,3005) 'Total cpu time needed for this calculation= ',(t2-t1) 
  write(2,3010) 'Riemann-Siegel NT value= ',RSN
  write(2,3009) 'Riemann-Sieg NT frac= ',rsfrac
  tsum=tsum+rstime
  do i=1,nchalf
   zsum(i)=zsum(i)+rszsum(i)

   !     Estimate for Z(t)=ZP(t)+RS part
   
   if (i.ge.numbercalc) then
    write(2,3008) 'Grand total of Hardy function Z(t+',(i-numbercalc)*0.01,')= ',zsum(i)
   else
    write(2,3008) 'Grand total of Hardy function Z(t-',(numbercalc-i)*0.01,')= ',zsum(i)
   endif   
  enddo 
  write(2,3005) 'Total cpu time needed for this calculation= ',tsum
  close(2,status='keep')

  !     Done

  STOP

  write(2,2011) 'Calculation hasnt- increase the value of totblock'

2001 FORMAT(A21,1X,I17)
2002 FORMAT(A14,1X,I4,A18,1X,I14,1X,A17,1X,I14)
2003 FORMAT(A23,2X,F15.9)
2004 FORMAT(A34,2X,F20.1)
2005 FORMAT(A32,2X,E13.6)
2006 FORMAT(I6,2X,F14.9)
2007 FORMAT(A31,1X,I17)
2008 FORMAT(A34,2X,F24.1,1X,A32,2X,E13.6)
2009 FORMAT(A38)
2010 FORMAT(A54,1X,I13)
2011 FORMAT(A74)
2012 FORMAT(A34,2X,I19,1X,A32,2X,E13.6)
2013 FORMAT(a36,1X,F21.4,2X,F21.4)
2014 format(i7,2x,f26.2,2x,e13.6)
2015 format(a12,1x,f21.4,2x,f21.4)
2016 format(a18,2x,i7)
2017 format(a34,2x,f21.4,2x,f21.4,2x,f21.4)  
2018 format(A24,1X,F24.1)      


3001 FORMAT(A17,E41.34)
3002 FORMAT(A23,F14.8,2X,A28,I4)
3003 FORMAT(A24,F26.2,2X,A37,I3)
3004 FORMAT(A18,1X,F17.10)
3005 FORMAT(A43,1X,E13.6)
3006 FORMAT(A38,1X,I21)
3007 FORMAT(A34,1X,F17.10)
3008 FORMAT(A34,F4.2,A4,F17.10)
3009 FORMAT(A22,1X,F29.23,1X,A6,I2)
3010 FORMAT(A25,1X,I21)

4001 FORMAT(A45)
4002 FORMAT(3X,F7.5)
4003 FORMAT(6X,A45)
4004 FORMAT(A4,F7.5)
4005 FORMAT(A3,A45)
4006 FORMAT(13X,I1)
!4007 FORMAT(I2)
4008 FORMAT(A12,E47.40)
!4009 FORMAT(E47.40)
  
!call MPI_Finalize(IERR)


contains

  !subroutine qsum(a,b,len,type)
    !integer, parameter :: dp = selected_real_kind(33)
    !real(dp),intent(in) :: a(len)
    !real(dp),intent(inout) :: b(len)
    !integer type,len
    !b=b+a
  !end subroutine qsum

end subroutine compute_hardy_fast





SUBROUTINE ERF(Z,N,ER)

  !     COMPUTES ERF(EXP(-i*PI/4)*Z) FOR Z REAL.

  !     IT USES JUST N TERMS ONLY.
  !     USES TAYLOR SERIES IF ABS(Z) NEAR 1, POWER SERIES IF ABS(Z)<<1 AND
  !     ASYMPTOTIC SERIES IF ABS(Z)>>1. THE COEFFICIENTS OF THESE SERIES ARE
  !     STORED IN CEF,ZSMAL AND ZLAR RESPECTIVELY. THREE TAYLOR SERIES ARE USED
  !     CENTRED ON Z=1, 1.5 AND 2. BELOW ABS(Z)<0.8 THE POWER SERIES ABOUT Z=0
  !     IS EMPLOYED AND ABOVE ABS(Z)>2.25 THE ASYMPTOTIC SERIES IS USED.

  implicit none

  integer, parameter :: dp = selected_real_kind(33) 

  integer       :: N,I
  real(dp)      :: Z,P,E,ZA,SP,TPP,TPM
  complex(kind=16)   :: ER,CEF(7,0:6),ZLAR(-1:6),ZSMAL(-1:6),V,EPI4

  COMMON/PARMS/ CEF,ZLAR,ZSMAL
  COMMON/PARS4/ P,SP,TPP,TPM,EPI4

  IF (N.GT.6) THEN
     WRITE(6,*) 'INCREASE NUMBER OF COEFFICIENTS IN SUBROUTINE ERF.'
     STOP
  ENDIF

  !     DO ALL MAIN CALCULATIONS ASSUMING Z>0.

  ZA=ABS(Z)
  IF (ZA.LT.1E-14) THEN
     ER=ZSMAL(-1)*ZA
     GOTO 1000
  ENDIF
  ER=(0.0,0.0)
  IF (ZA.GE.0.8.AND.ZA.LT.1.125) THEN

     !     ZA CLOSE TO 1, USE TAYLOR APPROX OF ERF(EXP(-I*PI/4)*1)
     !     TO GET ACCURATE SOLUTION

     E=ZA-1.0
     IF (ABS(E).GT.1E-14) THEN
        DO I=1,N
           ER=ER+CEF(1,I)*(E**I)
        ENDDO
     ENDIF
     ER=ER+CEF(1,0)
  ELSE IF (ZA.GE.1.125.AND.ZA.LT.1.375) THEN

     !     ZA CLOSE TO 1.25, USE TAYLOR APPROX OF ERF(EXP(-I*PI/4)*1.25)
     !     TO GET ACCURATE SOLUTION

     E=ZA-1.25
     IF (ABS(E).GT.1E-14) THEN
        DO I=1,N
           ER=ER+CEF(5,I)*(E**I)
        ENDDO
     ENDIF
     ER=ER+CEF(5,0)
  ELSE IF (ZA.GE.1.375.AND.ZA.LT.1.625) THEN

     !     ZA CLOSE TO 1.5, USE TAYLOR APPROX OF ERF(EXP(-I*PI/4)*1.5)
     !     TO GET ACCURATE SOLUTION


     E=ZA-1.5
     IF (ABS(E).GT.1E-14) THEN
        DO I=1,N
           ER=ER+CEF(3,I)*(E**I)
        ENDDO
     ENDIF
     ER=ER+CEF(3,0)
  ELSE IF (ZA.GE.1.625.AND.ZA.LT.1.875) THEN

     !     ZA CLOSE TO 1.75, USE TAYLOR APPROX OF ERF(EXP(-I*PI/4)*1.75)
     !     TO GET ACCURATE SOLUTION


     E=ZA-1.75
     IF (ABS(E).GT.1E-14) THEN
        DO I=1,N
           ER=ER+CEF(6,I)*(E**I)
        ENDDO
     ENDIF
     ER=ER+CEF(6,0)
  ELSE IF (ZA.GE.1.875.AND.ZA.LT.2.125) THEN

     !     ZA CLOSE TO 2.0, USE TAYLOR APPROX OF ERF(EXP(-I*PI/4)*2)
     !     TO GET ACCURATE SOLUTION


     E=ZA-2.0
     IF (ABS(E).GT.1E-14) THEN
        DO I=1,N
           ER=ER+CEF(2,I)*(E**I)
        ENDDO
     ENDIF
     ER=ER+CEF(2,0) 
  ELSE IF (ZA.GE.2.125.AND.ZA.LT.2.375) THEN

     !     ZA CLOSE TO 2.25, USE TAYLOR APPROX OF ERF(EXP(-I*PI/4)*2.25)
     !     TO GET ACCURATE SOLUTION


     E=ZA-2.25
     IF (ABS(E).GT.1E-14) THEN
        DO I=1,N
           ER=ER+CEF(7,I)*(E**I)
        ENDDO
     ENDIF
     ER=ER+CEF(7,0)

  ELSE IF (ZA.GE.2.375.AND.ZA.LT.2.625) THEN

     !     ZA CLOSE TO 2.5, USE TAYLOR APPROX OF ERF(EXP(-I*PI/4)*2.5)
     !     TO GET ACCURATE SOLUTION


     E=ZA-2.5
     IF (ABS(E).GT.1E-14) THEN
        DO I=1,N
           ER=ER+CEF(4,I)*(E**I)
        ENDDO
     ENDIF
     ER=ER+CEF(4,0)     
  ELSE IF (ZA.GE.2.625) THEN

     !    ZA LARGE, USE ASMPTOTIC APPROXIMATION. ACTUALLY WE COMPUTE
     !    ERFC(EXP(-i*PI/4)*Z) FIRST AND THEN USE ERF(Z)=1-ERFC(Z).
     !    THE APPROXIMATION IS VALID WHEN ABS(ARG(EXP(-i*PI/4)*Z))=PI/4,
     !    WHICH APPLIES WHEN Z>0, IS LESS THAN 3*PI/4. THE CASE Z<0 IS DEALT
     !    WITH AT THE END.

     E=ZA*ZA
     V=((1.0,0.0)*COS(E)+(0.0,1.0)*SIN(E))/ZA
     DO I=0,N
        ER=ER+ZLAR(I)/(E**I)
     ENDDO

     !    CONVERT FROM ERFC TO ERF

     ER=(1.0D0,0.0D0)-ZLAR(-1)*V*ER
  ELSE

     !    ZA SMALL<1, USE THE SERIES SOLUTION WHICH CONVERGES RAPIDLY.

     E=ZA*ZA
     DO I=0,N
        ER=ER+ZSMAL(I)*(E**I)
     ENDDO
     ER=ZA*ZSMAL(-1)*ER
  ENDIF

  !     NOW IF Z<0 ORIGINALLY USE ERF(-Z)=-ERF(Z)

1000 IF (Z.LT.0.0D0) THEN
     ER=-ER
  ENDIF
  RETURN
END SUBROUTINE ERF


SUBROUTINE LGAM(GA,XX)

  !     COMPUTES THE LOGARITHM OF THE GAMMA FUNCTION FOR REAL GA>0.
  !     IN THIS APPLICATION ONLY CALLED WHEN GA>12 AND FACTORIAL FUNCTION GETS LARGE VERY FAST.

  implicit none

  integer, parameter :: dp = selected_real_kind(33) 

  integer       :: I
  real(dp)      :: GA,XX,P,SP,TPP,TPM,GAMCOF(6),XXX,Y,TMP,SUM
  complex(kind=16)  :: EPI4

  COMMON/PARS4/ P,SP,TPP,TPM,EPI4
  COMMON/PARS8/ GAMCOF

  IF (ABS(GA).LT.1D-14) THEN
     XX=0.0
     RETURN
  ENDIF

  XXX=GA
  Y=XXX
  TMP=XXX+5.5D0
  TMP=(XXX+0.5D0)*LOG(TMP)-TMP
  SUM=1.000000000190015D0
  DO I=1,6
     Y=Y+1.0D0
     SUM=SUM+GAMCOF(I)/Y
  ENDDO

  XX=TMP+LOG(SQRT(TPP)*SUM/XXX)

  RETURN
END SUBROUTINE LGAM


subroutine ter(k,ain,yphase,t,term)

  implicit none

  integer, parameter :: dp = selected_real_kind(33)  
  integer, parameter :: dp1 = selected_int_kind(16)

  integer(dp1)  :: k
  real(dp)      :: ain,yphase,t,term,s,v,b

  s=k*ain
  b=sqrt(s*s-1)
  v=t*(b*(b-s)+log(s+b))+yphase
  term=cos(v)/sqrt(b)
  return
END subroutine ter




subroutine start(a,t6,yphase,zsum,M2)

  implicit none

  integer, parameter :: dp = selected_real_kind(33) 
  integer, parameter :: dp1 = selected_int_kind(16) 

  integer       :: itra,i
  integer(dp1)  :: M2
  real(dp)      :: a,t6,yphase,zsum,g,gm,p,sp,tpp,tpm,bot,top,t3,C,cc,AA,wb
  real(dp)      :: abb(64),wee(64),ri
  complex(kind=16) :: EPI4,sum,BB,expon,func,AAC
       
  COMMON/PARS4/ P,SP,TPP,TPM,EPI4
  
  itra=0
  if (mod(floor(a,dp1)+2,2).eq.0) then
     M2=floor(a,dp1)+1
  else
     M2=floor(a,dp1)+2
  endif
  g=(M2-a)*t6
  gm=(a-(M2-2))*t6
  zsum=0.0d0
  if (g.lt.3.2) then
     itra=1
     M2=M2+2
     C=1.0
     sum=(0.0,0.0)
     t3=t6*t6
     cc=tpp**0.25
     bot=-2*cc*sqrt(g)/t3
     AA=cc*(sqrt(2*g)*t3*t3*0.25-t3*g*cc)*0.5
     top=4/sqrt(AA)
     wb=t3*((0.25*t3-1.5*cc*sqrt(2*g))*t3+6.375*cc*cc*g)/(6*C)
     BB=EPI4*wb
        
     call gauleg(bot,top,abb,wee)

     do i=1,64
      expon=-(abb(i)**2)*(AA+abb(i)*BB)
      func=exp(expon)
      sum=sum+wee(i)*func
     enddo
      
     write(2,6001) '   Initial int = ',real(sum),aimag(sum)
        
     wb=yphase+0.5*sqrt(tpp)*g*t3-sqrt(p)*cc*(g**1.5)/(3*C)
     expon=(0.0,1.0)*wb
     ri=2*real(exp(expon)*sum)
     zsum=(M2-2)*ri/(sqrt(2*a)*(2**(1.5)))
     write(2,6000) '  Initial zsum = ',zsum
  endif
  if (gm.lt.3.2) then
     itra=-1
     M2=M2
     C=1.0
     sum=(0.0,0.0)
     t3=t6*t6
     cc=tpp**0.25
     bot=0.0
     AA=0.5*(sqrt(tpp)*gm*t3)
     AAC=AA+(0.0,1.0)*0.5*(cc*sqrt(2*gm)*t3*t3*0.25-1.75*(cc**3)*(gm**1.5)/sqrt(2*C))
     BB=(0.0,1.0)*1.5*cc*sqrt(2*gm)*t3*t3/(6*C)
     BB=(t3*(0.25*t3*t3-6.375*cc*cc*gm))/(6*C)-BB
     BB=EPI4*BB
     top=3/(real(BB)**(1/(3*C)))
                
     call gauleg(bot,top,abb,wee)

     do i=1,64
      expon=-(abb(i)**2)*(AAC+abb(i)*BB)
      func=exp(expon)
      sum=sum+wee(i)*func
     enddo
     write(2,6001) '   Initial int = ',real(sum),aimag(sum)

     wb=yphase-0.5*sqrt(tpp)*gm*t3
     expon=(0.0,1.0)*wb
     ri=2*real(exp(expon)*sum*exp(-sqrt(p)*cc*(gm**1.5)/(3*C)))
     zsum=(M2-2)*ri/(sqrt(2*a)*(2**(1.5)))
     write(2,6000) '  Initial zsum = ',zsum      
  endif
  return

6000   FORMAT(A17,2X,E38.31)       
6001   FORMAT(A17,2X,E38.31,2X,E38.31)  
END subroutine start




subroutine alphasum(MTM,rae,pcsum)

  implicit none

  integer, parameter :: dp = selected_real_kind(33)  
  integer, parameter :: dp1 = selected_int_kind(16) 

  integer       :: K,ip,irr,j,mc,im,i,jj
  integer(dp1)  :: MTM,JF,numbercalc,nchalf
  real(dp)      :: rae,pcsum(15),p,sp,tar(15),aar(15),et,e,s,amp(15),s1,pc,xx,th1,th2,wm(15),wp(15)
  real(dp)      :: rs,x0,ht,ain,yphasear(15),Y,pc2,con1,con2,tpp,tpm,initialcoeffcc(3),initialcoeff(3)
  real(dp)      :: ga1,ga2,ga3,cc,cc1,se,tse(15),pcc
  complex(kind=16)   :: c1,c2,pcgsum(1:2),EPI4,pha

  COMMON/PARS4/ P,SP,TPP,TPM,EPI4
  COMMON/PARS6/ tar,aar,ain,yphasear,et,ht,Y,numbercalc,K,ip,mc

  !       if (MTM.gt.2*K) then
  !        write(6,*) 'length of firstsums= ',MTM
  !        write(6,*) 'ae value= ',ae
  !       endif
  nchalf=2*numbercalc-1
  e=(rae*ain)**2
  s=sqrt(e-1.0)
  pc=2*(e+sqrt(e)*s)-1
  !       write(6,*) 'pc= ',pc
  pc2=(pc*pc-1)
!  s1=a/sqrt(pc)
!  amp=sqrt(8/(a*s))
  xx=1/(pc-1)
!  rs=p*pc/(2*pc2)
  con2=((sqrt(pc)*xx)**3)/(aar(numbercalc))

  ! first zeroth order phases, just I*wp and I*wm will do.

!  SE=(log(pc)+1/pc+1)/2.0


!  rae = fraction(rae)
!  print*, 'RAE = ', rae

  !AB_FLAG - rounding of rae occurs inside pari_tse.

  CALL pari_tse(rae,numbercalc,tse)

!  PRINT*, ' t*SE mod 2 pi = ', tse

!  STOP 

  

!  JF=AINT(SE,dp1)
  !  SE=SE-JF
  do i=1,nchalf
     e=(rae/aar(i))**2
     s=sqrt(e-1.0)
     pcc=2*(e+sqrt(e)*s)-1
     amp(i)=sqrt(8/(aar(i)*s))
     s1=aar(i)/sqrt(pcc)
     wp(i)=-tse(i)-p*(1-2*(xx+s1))/8.0
     wm(i)=wp(i)-p*s1/2.0
     wp(i)=wp(i)+p*con2/3.0
     wm(i)=wm(i)-p*con2/3.0
   enddo
  !       im=-1
  !       do j=3,mc
  !        cc=rs*(con2**(j-2))
  !        wp=wp+cc
  !        wm=wm+im*cc
  !        im=-im
  !       enddo


   ! now first order phases
   
  s1=aar(numbercalc)/sqrt(pc)
  initialcoeff(1)=xx/2.0+s1/4.0+con2
  initialcoeffcc(1)=initialcoeff(1)-s1/2.0-2*con2

  !      ga1=2.0
  !      ga1=lgamma(ga1)
  !      im=-1
  !      do j=3,mc
  !       ga2=(j+1)*1.0
  !       ga3=j*1.0
  !       cc=rs*(con2**(j-2))*exp(lgamma(ga2)-lgamma(ga3)-ga1)/p
  !       initialcoeff(1)=initialcoeff(1)+cc
  !       initialcoeffcc(1)=initialcoeffcc(1)+im*cc
  !       im=-im
  !      enddo
  initialcoeff(1)=initialcoeff(1)-anint(initialcoeff(1),dp1)
  initialcoeffcc(1)=initialcoeffcc(1)-anint(initialcoeffcc(1),dp1)


  !  now second order phases

  initialcoeff(2)=xx/2.0+2*con2
  initialcoeffcc(2)=initialcoeff(2)-4*con2

  !      ga1=3.0
  !      ga1=lgamma(ga1)
  !      im=-1
  !      do j=3,mc
  !       ga2=(j+1)*1.0
  !       ga3=(j-1)*1.0
  !       cc=rs*(con2**(j-2))*exp(lgamma(ga2)-lgamma(ga3)-ga1)*(2/p)
  !       initialcoeff(2)=initialcoeff(2)+cc
  !       initialcoeffcc(2)=initialcoeffcc(2)+im*cc
  !       im=-im
  !      enddo       


  !  now i=3,4,5,..mc phases

  do i=3,mc
     initialcoeff(i)=4*con2/3.0
     initialcoeffcc(i)=-initialcoeff(i)

     !       ga1=(i+1)*1.0
     !       ga1=lgamma(ga1)
     !       if (mod(i,2).lt.0.25) then
     !        im=1
     !       else
     !        im=-1
     !       endif
     !       do j=i,mc
     !        ga2=(j+1)*1.0
     !        ga3=(j-i+1)*1.0
     !        cc=rs*(con2**(j-2))*exp(lgamma(ga2)-lgamma(ga3)-ga1)*(2**(i-1))/p
     !        initialcoeff(i)=initialcoeff(i)+cc
     !        initialcoeffcc(i)=initialcoeffcc(i)+im*cc
     !        im=-im
     !       enddo
  enddo


  !      If sum length is short, then compute it directly.

  if (MTM.lt.K) then
     pcgsum(1)=(1.0,0.0)
     pcgsum(2)=pcgsum(1)
     do jj=1,MTM
        cc=jj*initialcoeff(mc)
        cc1=jj*initialcoeffcc(mc)
        do i=mc-1,1,-1
           cc=(cc+initialcoeff(i))*jj
           cc1=(cc1+initialcoeffcc(i))*jj
        enddo
        pha=(0.0,1.0)*tpp*cc
        pcgsum(1)=pcgsum(1)+exp(pha)
        pha=(0.0,1.0)*tpp*cc1
        pcgsum(2)=pcgsum(2)+exp(pha)
     enddo
  else

     call computarrs(mc,initialcoeff,MTM,K)
     call mgausssum(mc,ip,pcgsum(1))
     call computarrs(mc,initialcoeffcc,MTM,K)
     call mgausssum(mc,ip,pcgsum(2))


     !        if (abs(pcgsum(2)).gt.(15*abs(pcgsum(1)))) then
     !          write(2,*) rae,pcgsum(1),pcgsum(2)
     !         do i=1,3
     !         write(2,*) i,initialcoeff(i),initialcoeffcc(i)
     !        enddo
     !        write(2,*) pcgsum(1)
     !        write(2,*) pcgsum(2)
     !        endif
  endif

  do i=1,nchalf
   c1=(0.0,1.0)*wp(i)
   c1=exp(c1)
   c2=(0.0,1.0)*wm(i)
   c2=exp(c2)
   pcsum(i)=amp(i)*(real(c1*pcgsum(1)+c2*pcgsum(2)))
  enddo 
  return
END subroutine alphasum



!     THE NEXT ROUTINE COMPUTES THE PSI FUNCTION FOR REAL X

SUBROUTINE PSI(X,PS)

  IMPLICIT NONE

  integer, parameter :: dp = selected_real_kind(33) 

  real(dp) X,PS,P,GAM,ZET(3:13),XX,Z,SUM,CA,Q,SP,TPP,TPM
  complex(kind=16)   :: EPI4
  INTEGER  :: I

  COMMON/PARS2/ GAM,ZET
  COMMON/PARS4/ P,SP,TPP,TPM,EPI4

  XX=ABS(X)

  IF (XX.LT.1e-12) THEN
     WRITE(6,*) 'PSI FUNCTION UNDEFINED AT X=0'
     STOP
  ENDIF

  IF (X.LT.0.0.AND.ABS(X-NINT(X)).LT.1e-12) THEN
     WRITE(6,*) 'PSI FUNCTION UNDEFINED AT NEGATIVE INTEGERS'
     STOP
  ENDIF

  !      SOME AWKWARD SPECIAL VALUES PSI(1)=-GAM, PSI(2), PSI(3), PSI(0.5).

  IF (ABS(XX-1.0).LT.1e-12) THEN
     PS=-GAM
     IF (X.LT.0.0) THEN
        GOTO 4020
     ENDIF
     RETURN
  ENDIF

  IF (ABS(XX-2.0).LT.1e-12) THEN
     PS=0.422784335098467d0
     IF (X.LT.0.0) THEN
        GOTO 4020
     ENDIF
     RETURN
  ENDIF

  IF (ABS(XX-3.0).LT.1e-12) THEN
     PS=0.922784335098467d0
     IF (X.LT.0.0) THEN
        GOTO 4020
     ENDIF
     RETURN
  ENDIF

  !      PSI(1/2)=-GAM-2*LOG(2)

  IF (ABS(XX-0.5).LT.1e-12) THEN
     PS=-GAM-2*LOG(2.0)
     IF (X.LT.0.0) THEN
        GOTO 4020
     ENDIF
     RETURN
  ENDIF

  !      WORK WITH POSITIVE X VALUES VIZ. XX=ABS(X) FIRST. WORK OUT PSI(XX)

  IF (XX.LT.4.0) THEN

     !   magnitude of xx<4 use series solution valid for small xx. For positive xx=abs(x)

     IF (XX.LT.0.5) THEN
        Z=XX
        Q=-1/Z
     ELSE IF (XX.GT.0.5.AND.XX.LE.1.5) THEN
        Z=XX-1.0
        Q=0.0
     ELSE IF (XX.GT.1.5.AND.XX.LE.2.5) THEN
        Z=XX-2.0
        Q=1/(Z+1.0)
     ELSE IF (XX.GT.2.5.AND.XX.LE.3.5) THEN
        Z=XX-3.0   
        Q=1/(Z+1.0)+1/(Z+2.0)
     ELSE
        Z=XX-4.0
        Q=1/(Z+1.0)+1/(Z+2.0)+1/(Z+3.0)
     ENDIF

     !   series for psi(1+z) equation 5.7.5. psi(2+z)=psi(1+z)+1/(1+z); psi(3+z)=psi(1+z)+1/(1+z)+1/(2+z); with z an element [-0.5,0.5].

     SUM=0.0
     DO I=1,6
        SUM=SUM+(ZET(2*I+1)-1.0)*(Z**(2*I))
     ENDDO
     IF (ABS((Z-0.5)-NINT(Z-0.5)).LT.1e-12) THEN
        CA=0.0
     ELSE
        CA=0.5*P/TAN(P*Z)
     ENDIF
     PS=1/(2*Z)-CA+1/(Z*Z-1.0)+1-GAM-SUM+Q
  ELSE

     ! magnitude of xx>4 asymptotic series converges fast

     PS=LOG(XX)-0.5/XX-1/(12.0*XX*XX)+1/(120.0*(XX**4))-1/(252.0*(XX**6))+1/(240.0*(XX**8))
  ENDIF

  !  if x<0 then use reflection formula

4020 IF (X.LT.0.0)THEN
     IF (ABS((XX-0.5)-NINT(XX-0.5)).LT.1e-12) THEN
        CA=0.0
     ELSE
        CA=P/TAN(P*XX)
     ENDIF
     PS=PS+1/XX+CA
  ENDIF
  RETURN
END SUBROUTINE PSI



subroutine scheme(m,j,small)

  implicit none

  integer, parameter :: dp = selected_real_kind(33)
  integer, parameter :: dp1 = selected_int_kind(16)    

  integer       :: m,j,icj(0:50),MIT,i,ki,mmax
  integer(dp1)  :: L(0:50)
  real(dp)      :: small,phicoeff(0:10,50),xr(50),fracL(0:50),see(10),x,y
  real(dp)      :: rfact1,con1,rfact2,con2,rfact3,con3,con4

  COMMON/PARS1/ phicoeff,fracL,xr,L,icj,MIT,mmax
  COMMON/PARS7/ see

  y=2*phicoeff(2,j)
  see(1)=1/(2*y)
  see(2)=-phicoeff(3,j)/(y**3)
  ki=1
  rfact1=0.5
  rfact2=1.0
  rfact3=2.0
  do i=3,m
     if (i.eq.3) then
        small=9*(phicoeff(3,j)**2)/(2*(y**5))
        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     else if (i.eq.4) then
        rfact1=rfact1*(2*i-3)*6/((i+1)*1.0)
        con1=(phicoeff(3,j)**(i-3))*(9*(phicoeff(3,j)**2)-2*(i-2)*y*phicoeff(4,j))
        small=rfact1*con1/(y**7)
        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     else if (i.eq.5) then       
        rfact1=rfact1*(2*i-3)*6/((i+1)*1.0)
        con1=(phicoeff(3,j)**(i-3))*(9*(phicoeff(3,j)**2)-2*(i-2)*y*phicoeff(4,j))
        small=rfact1*con1/(y**9)

        con2=(phicoeff(3,j)**(i-5))*(15*phicoeff(3,j)*phicoeff(5,j)+8*(i-4)*(phicoeff(4,j)**2))
        small=small+rfact2*con2/(y**7)   

        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     else if (i.eq.6) then       
        rfact1=rfact1*(2*i-3)*6/((i+1)*1.0)
        con1=(phicoeff(3,j)**(i-3))*(9*(phicoeff(3,j)**2)-2*(i-2)*y*phicoeff(4,j))
        small=rfact1*con1/(y**11)

        rfact2=rfact2*6*(i-2)*(2*i-5)/((i+1)*1.0*(i-4))
        con2=(phicoeff(3,j)**(i-5))*(15*phicoeff(3,j)*phicoeff(5,j)+8*(i-4)*(phicoeff(4,j)**2))
        small=small+rfact2*con2/(y**9)

        con3=(phicoeff(3,j)**(i-6))*(9*phicoeff(3,j)*phicoeff(6,j)+10*(i-5)*phicoeff(4,j)*phicoeff(5,j) &
             +(i-6)*(i-5)*16*(phicoeff(4,j)**3)/(9*phicoeff(3,j)))
        small=small-rfact3*con3/(y**8)

        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     else if (i.eq.7) then       
        rfact1=rfact1*(2*i-3)*6/((i+1)*1.0)
        con1=(phicoeff(3,j)**(i-3))*(9*(phicoeff(3,j)**2)-2*(i-2)*y*phicoeff(4,j))
        small=rfact1*con1/(y**13)

        rfact2=rfact2*6*(i-2)*(2*i-5)/((i+1)*1.0*(i-4))
        con2=(phicoeff(3,j)**(i-5))*(15*phicoeff(3,j)*phicoeff(5,j)+8*(i-4)*(phicoeff(4,j)**2))
        small=small+rfact2*con2/(y**11)

        rfact3=rfact3*6*(i-3)*(2*i-5)/((i+1)*1.0*(i-4))
        con3=(phicoeff(3,j)**(i-6))*(9*phicoeff(3,j)*phicoeff(6,j)+10*(i-5)*phicoeff(4,j)*phicoeff(5,j) &
             +(i-6)*(i-5)*16*(phicoeff(4,j)**3)/(9*phicoeff(3,j)))
        small=small-rfact3*con3/(y**10)

        small=small+(42*phicoeff(3,j)*phicoeff(7,j)+48*phicoeff(4,j)*phicoeff(6,j)+25*(phicoeff(5,j)**2)) &
             /(2*(y**9))

        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     else if (i.eq.8) then       
        rfact1=rfact1*(2*i-3)*6/((i+1)*1.0)
        con1=(phicoeff(3,j)**(i-3))*(9*(phicoeff(3,j)**2)-2*(i-2)*y*phicoeff(4,j))
        small=rfact1*con1/(y**15)

        rfact2=rfact2*6*(i-2)*(2*i-5)/((i+1)*1.0*(i-4))
        con2=(phicoeff(3,j)**(i-5))*(15*phicoeff(3,j)*phicoeff(5,j)+8*(i-4)*(phicoeff(4,j)**2))
        small=small+rfact2*con2/(y**13)

        rfact3=rfact3*6*(i-3)*(2*i-5)/((i+1)*1.0*(i-4))
        con3=(phicoeff(3,j)**(i-6))*(9*phicoeff(3,j)*phicoeff(6,j)+10*(i-5)*phicoeff(4,j)*phicoeff(5,j) &
             +(i-6)*(i-5)*16*(phicoeff(4,j)**3)/(9*phicoeff(3,j)))
        small=small-rfact3*con3/(y**12)

        small=small+5*(63*(phicoeff(3,j)**2)*phicoeff(7,j)+144*phicoeff(3,j)*phicoeff(4,j)*phicoeff(6,j) &
             +75*phicoeff(3,j)*(phicoeff(5,j)**2)+80*(phicoeff(4,j)**2)*phicoeff(5,j))/(y**11)

        small=small-2*(12*phicoeff(3,j)*phicoeff(8,j)+14*phicoeff(4,j)*phicoeff(7,j)+15*phicoeff(5,j)*phicoeff(6,j))/(y**10)

        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     else if (i.eq.9) then       
        rfact1=rfact1*(2*i-3)*6/((i+1)*1.0)
        con1=(phicoeff(3,j)**(i-3))*(9*(phicoeff(3,j)**2)-2*(i-2)*y*phicoeff(4,j))
        small=rfact1*con1/(y**17)

        rfact2=rfact2*6*(i-2)*(2*i-5)/((i+1)*1.0*(i-4))
        con2=(phicoeff(3,j)**(i-5))*(15*phicoeff(3,j)*phicoeff(5,j)+8*(i-4)*(phicoeff(4,j)**2))
        small=small+rfact2*con2/(y**15)

        rfact3=rfact3*6*(i-3)*(2*i-5)/((i+1)*1.0*(i-4))
        con3=(phicoeff(3,j)**(i-6))*(9*phicoeff(3,j)*phicoeff(6,j)+10*(i-5)*phicoeff(4,j)*phicoeff(5,j) &
             +(i-6)*(i-5)*16*(phicoeff(4,j)**3)/(9*phicoeff(3,j)))
        small=small-rfact3*con3/(y**14)

        small=small+11*(378*(phicoeff(3,j)**3)*phicoeff(7,j)+1296*(phicoeff(3,j)**2)*phicoeff(4,j)*phicoeff(6,j)+675* &
             ((phicoeff(3,j)*phicoeff(5,j))**2)+1440*phicoeff(3,j)*(phicoeff(4,j)**2)*phicoeff(5,j)+128*(phicoeff(4,j)**4))/(y**13)

        small=small-22*(18*(phicoeff(3,j)**2)*phicoeff(8,j)+42*phicoeff(3,j)*phicoeff(4,j)*phicoeff(7,j)+45*phicoeff(3,j)* &
             phicoeff(5,j)*phicoeff(6,j)+24*(phicoeff(4,j)**2)*phicoeff(6,j)+25*phicoeff(4,j)*(phicoeff(5,j)**2))/(y**12)

        small=small+(27*phicoeff(3,j)*phicoeff(9,j)+32*phicoeff(4,j)*phicoeff(8,j)+35*phicoeff(5,j)*phicoeff(7,j)+18* &
             (phicoeff(6,j)**2))/(y**11)

        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     else if (i.eq.10) then       
        rfact1=rfact1*(2*i-3)*6/((i+1)*1.0)
        con1=(phicoeff(3,j)**(i-3))*(9*(phicoeff(3,j)**2)-2*(i-2)*y*phicoeff(4,j))
        small=rfact1*con1/(y**19)

        rfact2=rfact2*6*(i-2)*(2*i-5)/((i+1)*1.0*(i-4))
        con2=(phicoeff(3,j)**(i-5))*(15*phicoeff(3,j)*phicoeff(5,j)+8*(i-4)*(phicoeff(4,j)**2))
        small=small+rfact2*con2/(y**17)

        rfact3=rfact3*6*(i-3)*(2*i-5)/((i+1)*1.0*(i-4))
        con3=(phicoeff(3,j)**(i-6))*(9*phicoeff(3,j)*phicoeff(6,j)+10*(i-5)*phicoeff(4,j)*phicoeff(5,j) &
             +(i-6)*(i-5)*16*(phicoeff(4,j)**3)/(9*phicoeff(3,j)))
        small=small-rfact3*con3/(y**16)

        con4=273*(189*(phicoeff(3,j)**3)*phicoeff(7,j)+864*(phicoeff(3,j)**2)*phicoeff(4,j)*phicoeff(6,j)+450* &
             ((phicoeff(3,j)*phicoeff(5,j))**2)+1440*phicoeff(3,j)*(phicoeff(4,j)**2)*phicoeff(5,j)+256*(phicoeff(4,j)**4))
        small=small+con4*phicoeff(3,j)/(y**15)

        con4=108*(phicoeff(3,j)**3)*phicoeff(8,j)+378*(phicoeff(3,j)**2)*phicoeff(4,j)*phicoeff(7,j)
        con4=con4+405*(phicoeff(3,j)**2)*phicoeff(5,j)*phicoeff(6,j)+432*(phicoeff(4,j)**2)*phicoeff(3,j)*phicoeff(6,j)
        con4=con4+450*(phicoeff(5,j)**2)*phicoeff(3,j)*phicoeff(4,j)+160*(phicoeff(4,j)**3)*phicoeff(5,j)
        small=small-52*con4/(y**14)

        con4=243*(phicoeff(3,j)**2)*phicoeff(9,j)+576*phicoeff(3,j)*phicoeff(4,j)*phicoeff(8,j)
        con4=con4+630*phicoeff(3,j)*phicoeff(5,j)*phicoeff(7,j)+324*(phicoeff(6,j)**2)*phicoeff(3,j)
        con4=con4+336*(phicoeff(4,j)**2)*phicoeff(7,j)+720*phicoeff(4,j)*phicoeff(5,j)*phicoeff(6,j)+125*(phicoeff(5,j)**3)
        small=small+2*con4/(y**13)

        con4=(30*phicoeff(3,j)*phicoeff(10,j)+36*phicoeff(4,j)*phicoeff(9,j)+40*phicoeff(5,j)*phicoeff(8,j)  &
             +42*phicoeff(6,j)*phicoeff(7,j))
        small=small-con4/(y**12)

        if (i.lt.m) then
           see(i)=ki*small-phicoeff(i+1,j)/(y**(i+1))
        else 
           see(i)=ki*small
        endif
        ki=ki*(-1)
     endif
  enddo

  return
end subroutine scheme

subroutine phifactors(m,j)

  implicit none

  integer, parameter :: dp = selected_real_kind(33)
  integer, parameter :: dp1 = selected_int_kind(16)    

  integer       :: m,j,icj(0:50),MIT,i,nk,n,if,mmax
  integer(dp1)  :: L(0:50)
  real(dp)      :: phicoeff(0:10,50),xr(50),fracL(0:50),see(10),x,y,sum,y1,y2,GA

  COMMON/PARS1/ phicoeff,fracL,xr,L,icj,MIT,mmax
  COMMON/PARS7/ see

  x=-phicoeff(1,j)
  do i=0,m
     if (i.eq.0.or.i.eq.1) then
        nk=2
     else
        nk=i
     endif
     sum=0.0d0

     !        for large m uses the fact that m!=gamma(m+1) and works out log(m!)=log(gamma(m+1)) and then
     !        combines logs to get combinatorial function n!/((n-i)!i!).

     if (m.gt.12) then
        ga=(i+1)*1.0
        call lgam(ga,y)
        do n=nk,m
           ga=(n+1)*1.0
           call lgam(ga,y1)
           ga=(n-i+1)*1.0
           call lgam(ga,y2)
           sum=sum+exp(y1-y2-y)*(x**(n-i))*see(n-1)
        enddo
     else

        !        for small m just work out factorials long hand

        y=1.0
        do if=2,i
           y=y*if
        enddo
        do n=nk,m
           y1=1.0
           do if=2,n
              y1=y1*if
           enddo
           y2=1.0
           do if=2,(n-i)
              y2=y2*if
           enddo
           sum=sum+y1*(x**(n-i))*see(n-1)/(y*y2)
        enddo
     endif
     phicoeff(i,j+1)=sum
     if (i.gt.2.and.abs(phicoeff(i,j+1)).gt.0.5) then
        phicoeff(i,j+1)=phicoeff(i,j+1)-anint(phicoeff(i,j+1),dp1)
     endif
  enddo

  return
end subroutine phifactors


subroutine computecnewton(n,nit,con1,tol,c)

  !       Computes value of saddle point c for large dimensional schemes.
  !       Specifically it evaluate c such that
  !
  !       f(c)=(m-phicoeff(1,nit))-Sum[q=2..n] of q*phicoeff(q,nit)*(c^(q-1))=0
  !
  !       for given integer value. Uses standard 1D Newton-Raphson to a tolerance tol>0
  !       con1=(m-phicoeff(1)), n=scheme value only used for m>=6. On entry c is a guess of the answer.
  !       On exit c=final answer.

  implicit none

  integer, parameter :: dp = selected_real_kind(33)
  integer, parameter :: dp1 = selected_int_kind(16) 

  integer       :: n,nit,icj(0:50),MIT,itermax,i,j,mmax
  integer(dp1)  :: L(0:50)
  real(dp)      :: con1,tol,c,cold,sum1,sum2
  real(dp)      :: phicoeff(0:10,50),xr(50),fracL(0:50)


  COMMON/PARS1/ phicoeff,fracL,xr,L,icj,MIT,mmax

  cold=c
  itermax=5
  do i=1,itermax
     sum1=con1
     sum2=-2*phicoeff(2,nit)
     do j=2,n
        sum1=sum1-j*phicoeff(j,nit)*(cold**(j-1))
        if (j.gt.2) then
           sum2=sum2-j*(j-1)*phicoeff(j,nit)*(cold**(j-2))
        endif
     enddo
     c=cold-sum1/sum2
     if (abs(c-cold).lt.tol) then
        return
     else
        cold=c
     endif
  enddo
  write(6,*) 'no convergence in computecnewton after 5 iterations'
  stop

end subroutine computecnewton






subroutine mgausssum(m,ip,csum)

  implicit none

  integer, parameter :: dp = selected_real_kind(33)  
  integer, parameter :: dp1 = selected_int_kind(16)

  integer       :: m,ip,MIT,icj(0:50),i,k,j,m1(50),jj,mmax
  integer(dp1)  :: L(0:50)
  real(dp)      :: initialcoeff(3),errtol,p,sp,tpp,tpm
  real(dp)      :: phicoeff(0:10,50),xr(50),fracL(0:50),x,y,rminorcor,denn,rminorcor2,h1,h2,h3
  complex(kind=16)   :: csum,EPI4,fn,pp,c1,qq

  COMMON/PARS1/ phicoeff,fracL,xr,L,icj,MIT,mmax
  COMMON/PARS4/ P,SP,TPP,TPM,EPI4
  COMMON/PARS5/ m1


  if (L(MIT-1).eq.0.or.L(MIT-1).eq.(-1)) then
     csum=(1.0,0.0)
  else
     if (MIT.gt.1) then
        h1=3*phicoeff(3,MIT-1)/(xr(MIT-1)**2)
        h3=h1*h1
        if (m1(MIT-1).gt.3) then
           h2=phicoeff(4,MIT-1)/(xr(MIT-1)**3)
        else
           h2=0.0
        endif   
        denn=1-phicoeff(1,MIT-1)*(h1-phicoeff(1,MIT-1)*(6*h2-1.5*h3))
        rminorcor2=1.5*h3-6*h2
        rminorcor=h1+3*phicoeff(1,MIT-1)*(h3-4*h2)
     endif      
     csum=(0.0,0.0)
     do i=0,L(MIT-1)
        fn=(0.0,1.0)*tpm*i*(phicoeff(1,MIT)+i*(phicoeff(2,MIT)+i*phicoeff(3,MIT)))
        if (m1(MIT).gt.m) then
           y=i*1.0
           do j=(m+1),m1(MIT)
              fn=fn+(0.0,1.0)*tpm*(y**j)*phicoeff(j,MIT)
           enddo
        endif
        if (MIT.gt.1) then
           csum=csum+exp(fn)/(denn+i*(rminorcor-i*rminorcor2))
        else   
           csum=csum+exp(fn)
        endif   
     enddo
     if (MIT.gt.1) then
      if (phicoeff(1,MIT-1).gt.0.0) then
         csum=csum-1.0
      endif   
     endif
  endif
  if (icj(MIT).eq.(-1)) then
     csum=conjg(csum)
  endif
  do k=(MIT-1),1,-1
     x=abs(xr(k))
     pp=(0.0,1.0)*tpp*phicoeff(0,k+1)  
     c1=exp(pp)/(EPI4*sqrt(x))
     call q(m,k,ip,qq)
     csum=c1*csum+qq
     if (icj(k).eq.-1) then
        csum=conjg(csum)
     endif
     if (k.gt.1) then
       if (phicoeff(1,k-1).gt.0.0) then
          csum=csum-1.0
       endif   
     endif
  enddo

  return

end subroutine mgausssum



subroutine computarrs(m,initialcoeff,N,KCUT)

  implicit none

  integer, parameter :: dp = selected_real_kind(33)  
  integer, parameter :: dp1 = selected_int_kind(16)


  integer       :: m,KCUT,icj(0:50),ix,j,jj,MIT,its,m1(50),ichange,mmax
  integer(dp1)  :: N,L(0:50)
  real(dp)      :: initialcoeff(3),phicoeff(0:10,50),xr(50),fracL(0:50),x0,sum
  real(dp)      :: etaa,small,smalle,cons1,smalle1

  COMMON/PARS1/ phicoeff,fracL,xr,L,icj,MIT,mmax
  COMMON/PARS5/ m1



  m1(1)=m
  if (N.lt.1) then
     write(6,*) 'N is too small for computarrs'
     return
  endif

  !   Initial setup and first iteration

  L(0)=N     
  icj(0)=1
  x0=2*initialcoeff(2)
  ix=nint(x0)
  xr(1)=x0-ix
  phicoeff(2,1)=xr(1)/2.0
  if (mod(ix,2).eq.1) then
     if (initialcoeff(1).lt.0.0) then
        phicoeff(1,1)=initialcoeff(1)+0.5
     else
        phicoeff(1,1)=initialcoeff(1)-0.5
     endif
  else
     phicoeff(1,1)=initialcoeff(1)
  endif
  do jj=3,m
     phicoeff(jj,1)=initialcoeff(jj)
  enddo
  if (xr(1).gt.0.0) then
     icj(1)=-1
  else
     icj(1)=1
     phicoeff(1,1)=-phicoeff(1,1)
     xr(1)=abs(xr(1))
     phicoeff(2,1)=abs(phicoeff(2,1))
     do jj=3,m
        phicoeff(jj,1)=-phicoeff(jj,1)
     enddo
  endif

  sum=0.0
  etaa=L(0)*1.0
  do jj=1,m
     sum=sum+jj*phicoeff(jj,1)*(etaa**(jj-1))
  enddo
  L(1)=floor(sum,dp1)
  fracL(1)=sum-L(1)*1.0
  if (phicoeff(1,1).gt.0.0) then
     etaa=1.0-phicoeff(1,1)
  else
     etaa=1.0+phicoeff(1,1)
  endif

  !     Extra check. If sum very close to 1 on first iteration this is likely to lead to significant errors
  !     i.e. sum>1 but fracL(1)<<1 then setting L(1)=1 is unlikely to be good. So instead set it to L(1)=0
  !     and just do a full sum with no iterations to be on the safe side.

  if (L(1).eq.1.and.abs(kcut*fracL(1)).lt.1.0) then
     L(1)=0
  endif
  if ((L(1).eq.0).or.(L(1).eq.-1)) then
     !        xr(2)=0.0
     !        icj(2)=1
     !        do jj=1,m
     !         phicoeff(jj,2)=0.0
     !        enddo
     MIT=1
     !         write(6,1005) '  Final M= ',MIT
     return
  endif

  !  Main iteration loop

  do j=2,50
     ichange=0
     m1(j)=m1(j-1)
     call scheme(m1(j-1),j-1,small)
     call phifactors(m1(j-1),j-1)
984  ix=nint(2*phicoeff(2,j))
     xr(j)=2*phicoeff(2,j)-ix
     phicoeff(2,j)=0.5*xr(j)
     phicoeff(1,j)=phicoeff(1,j)-anint(phicoeff(1,j),dp1)
     if (mod(abs(ix),2).eq.1) then
        if (phicoeff(1,j).lt.0.0) then
           phicoeff(1,j)=phicoeff(1,j)+0.5
        else
           phicoeff(1,j)=phicoeff(1,j)-0.5
        endif
     endif
     if (xr(j).gt.0.0) then
        icj(j)=-1
     else
        xr(j)=-xr(j)
        icj(j)=1
        do jj=1,m1(j)
           phicoeff(jj,j)=-phicoeff(jj,j)
        enddo
     endif
     if (L(j-1).le.0) then
        L(j)=-2
     else
        sum=0.0
        etaa=L(j-1)*1.0
        do jj=1,m1(j)
           sum=sum+jj*phicoeff(jj,j)*(etaa**(jj-1))
        enddo
        L(j)=floor(sum,dp1)
        fracL(j)=sum-L(j)*1.0
        if (phicoeff(1,j).gt.0.0) then
           etaa=1.0-phicoeff(1,j)
        else
           etaa=1.0+phicoeff(1,j)
        endif
     endif
     x0=(L(j-1)-phicoeff(1,j-1))
     smalle=(x0**(m1(j)+1))*small
     if (abs(smalle).gt.0.001) then
        m1(j)=m1(j)+1
        if (m1(j).gt.mmax) then
           MIT=j-1
           !          write(6,1005) '  Final M= ',MIT
           return
        endif
        phicoeff(m1(j),j-1)=0.0
        call phifactors(m1(j),j-1)
        call scheme(m1(j),j-1,small)
        goto 984
     endif

     if (L(j).le.KCUT.and.L(j-1).le.KCUT) then
        MIT=j
        !          if (L(MIT-1).eq.ceiling(phicoeff(1,MIT-1))) then
        if (L(MIT-1).lt.6.and.MIT.ge.3) then
           MIT=MIT-1
        endif

        !     Extra check. For very large t the kernel sum could be be made up of
        !     double saddles. Such scenarios are not accounted for in this code and
        !     must be avoided at all costs by pushing the kernel sum one place back
        !     up the interation chain.
        
        if (phicoeff(3,MIT-1).lt.0.0) then
         if ((phicoeff(2,MIT-1)+3*phicoeff(3,MIT-1)*L(MIT-2)).lt.0.0) then
          MIT=MIT-1
         endif
        endif        
       return
     endif
     if (L(j).le.0.0) then
        MIT=j
        !          write(6,1005) '  Final M= ',MIT
        return
     endif
     if (L(j).gt.L(j-1)) then
        MIT=j
        !          write(6,1005) '  Final M= ',MIT
        return
     endif
  enddo
  write(6,*) 'no convergence after 50 iterations'

1005 format(A11,I2)

end subroutine computarrs





subroutine q(m,nit,ip,qq)

  implicit none

  integer, parameter :: dp = selected_real_kind(33)
  integer, parameter :: dp1 = selected_int_kind(16) 

  integer       :: m,nit,ip,i,j,ip1,icj(0:50),MIT,m1(50),jbot,mmax
  integer(dp1)  :: L(0:50)
  real(dp)      :: p,sp,tpp,tpm,e1,e2,e3,sum,ps1,ps2
  real(dp)      :: phicoeff(0:10,50),xr(50),fracL(0:50),sx,xcube,con1,con2,con3,z,wm,c,beta
  real(dp)      :: xbeta,c1,ecor,gc,sav1,sav2,tol
  complex(kind=16)   :: qq,epi4,t1,t2,t3,t4,t5,con5,endpoint
  complex(kind=16)   :: cr1,cr2,cr3,fn,sum0,sum1,sum2,sum3,sa(3),sb(3)

  COMMON/PARS1/ phicoeff,fracL,xr,L,icj,MIT,mmax
  COMMON/PARS4/ P,SP,TPP,TPM,EPI4
  COMMON/PARS5/ m1

  !      COMPUTES THE CORRECTION TERM QQ IN THE RECURSIVE SERIES

  !       GENERAL GAUSS SUM LENGTH L(NIT-1) = MULTIPLICATIVE FACTOR*(NEW GENERAL GAUSS SUM LENGTH L(NIT)+QQ

  !      WHERE L(NIT)~ABS(xr(nit))*L(NIT-1) WITH ABS(x(nit))<0.5. RECURSIVELY APPLIED IT CAN BE USED TO ESTIMATE 
  !      A VERY LONG GAUSS SUM (TRILLIONS UPON TRILLIONS OF SUMMANDS) FROM A RELATIVELY SHORT ONE (HUNDREDS OF SUMMANDS)
  !      L(0)>L(1)>L(2)....>L(MIT). NB the first sum is of length L(0) and coefficients phicoeff(1 to m,1), the nit sum
  !      is of length L(nit-1) and coefficients phicoeff(1 to m, nit) etc.

  !      Inputs
  !       m: classification of general gauss sum, m=3 cubic, m=4 quartic etc
  !       nmax: used to calulate wm factor. Defined to be largest integer
  !              less than (m+1-q)/l   m>=q>=2,  1<=l<=q. For m=3 nmax=3.
  !       nit: current iteration value, lies between 1 and MIT.
  !       ip: >=1 endpoint factor. Certain integrals derived from terms near the beginning and end of each
  !           Gauss sum are not asymptotically small in the way as the same integrals derived from the main
  !           bulk of the terms. These integrals near the endpoints must be computed and summed. The sums
  !           converge but relatively slowly, so ip must be large enough to ensure the sums have converged
  !           close enough to their actual values. I have found ip~20 to be adequate.
  !      Output
  !       qq:  The correction factor to the recursive scheme. It is made up of 5 sub-terms I have denoted
  !       t1-5. These terms are derived from integrals that appear in the Euler-Maclaurin summation formula
  !       which forms the crux of the recursive scheme.

  !       xcube=(xr(nit)**3)
  !       e1=-phicoeff(1,nit)/xr(nit)-3*(phicoeff(1,nit)**2)*phicoeff(3,nit)/xcube
  !       e2=1/(2*xr(nit))+3*phicoeff(1,nit)*phicoeff(3,nit)/xcube
  !       e3=-phicoeff(3,nit)/xcube

  !      e1,e2,e3 are the first 3 coefficients of the shorter Gauss sum on the RHS of the equation
  !      derived from the corresponding coefficients of the longer sum phicoeff(1-3,nit). These are used
  !      in the calculation of term t5.

  !       fn=(0.0,1.0)*tpp*(phicoeff(1,nit)**2)*(0.5/xr(nit)-phicoeff(1,nit)*e3)
  !       con5=exp(fn)

  !      con5 is actually the muliplicative factor in the recursive scheme. Also used for t5

  tol=1e-7
  sx=sqrt(xr(nit))
  sum=0.0

  e3=(L(nit-1)*1.0)
  do i=1,m1(nit)
     sum=sum+phicoeff(i,nit)*(e3**i)
  enddo
  fn=(0.0,1.0)*tpm*sum
  endpoint=exp(fn)
  t3=0.5*(1.0+endpoint)

  !      t3 easiest term to calculate, given by 0.5*(terms at k=0 and k=L(nit-1)) that appear in E-M summation formula.

  con1=1.0-fracL(nit)
  con2=L(nit)+fracL(nit)+1.0
  call psi(con2,ps2)
  call psi(con1,ps1)
  sav1=ps1
  con2=ps2-ps1

  !      ignore psi(2,z) correction here, which is included in MAPLE version

  call psi(phicoeff(1,nit)+1.0,ps2)
  call psi(L(nit)+1.0-phicoeff(1,nit),ps1)
  sav2=ps1
  con3=ps2-ps1
  t1=(0.0,1.0)*(conjg(endpoint)*con2-con3)/tpp

  !      t1 is the contribution from the int[0..L(nit-1)] psi(x)*f'(x)*exp(2*p*I*f(x))dx that doesn't
  !      involve saddles. PSi(x)=-1 sum[k=1 to infinity)]sin(2*p*k*x)/(p*k)

  z=sp*con1/sx
  call erf(z,6,cr1)
  cr2=1.0-cr1
  fn=-(0.0,1.0)*p*(con1**2)/xr(nit)
  cr3=p*exp(fn)*cr2/(sx*EPI4)-1/con1-1/(L(nit)*1.0+1.0)
  t2=(0.0,1.0)*conjg(endpoint)*cr3/tpp

  !      t2 acts as a correction to t1. t1 is derived using an asymptotic expansion for the erfc(z), with z large.
  !      Normally this is good enough, but if phicoeff(1,nit) is very near zero, or fracL(nit) very near zero it
  !      means that z is not really large at the first and last points to apply the asymptotic expansion.
  !      Effectively t2 knocks out the inaccurate asmptotic terms used for erfc(z) in t1 and replaces them with their
  !      exact erfc(z) values. Usually abs(t2) is relatively small, indicating that the asymptotic terms are pretty good.

  if (phicoeff(1,nit).gt.0.0) then
     con1=phicoeff(1,nit)
  else
     con1=phicoeff(1,nit)+1.0
  endif
  z=sp*con1/sx
  call erf(z,6,cr1)
  cr2=1.0-cr1
  fn=-(0.0,1.0)*p*(con1**2)/xr(nit)
  cr3=(0.0,1.0)*exp(fn)*cr2/(2*sx*EPI4)
  if (phicoeff(1,nit).gt.0.0) then
     t4=-(0.0,1.0)*conjg(endpoint)/(tpp*(L(nit)+fracL(nit)))+cr3
  else
     t4=(0.0,1.0)*(phicoeff(1,nit)/(con1)-1.0)/tpp+cr3
  endif

  !      If phicoeff(1,nit)>0, there is an integral that does not contain a saddle which must be estimated.
  !      This is the first correction made in t4. In this case the new sum starts at k=1,2,...L(nit).
  !      If phicoeff(1,nit)<0, the integral has a saddle at k=0 effectively. This saddle term acts as the first point
  !      of the new shortened sum k=0,1... L(nit). In addition to this zeroth saddle, there is a correction that is 
  !      included in t4. Overall the abs(t4) is usually small compared to t1,t3 and t5.

  !      Next is t5. This is the most difficult part of the correction term to find.
  !      Essentially these are second order terms associated with integrals over k with saddles.
  !      The contribution at the saddle is the first order correction and included in the new shortened sum.
  !      Most second order terms are tiny in comparison with their first order counterparts and can be ignored.
  !      But those close to the endpoints k=0,1,2  and k=L(nit),L(nit)-1,... are close to being the same size
  !      as the first order terms (because the saddle is situated very close to an endpoint of the integration
  !      range. So at both endpoints these second order tems have computed and sums. The value of ip sets how
  !      many are summed at each endpoint. Unfortunately convergence is slow so potentially ip can be large.
  !      I have got quite good results for 5<ip<30, but deciding on the best value is not obvious.


  sum0=(0.0,0.0)
  jbot=ceiling(phicoeff(1,nit))

  con1=jbot-phicoeff(1,nit)
  call psi(con1,ps1)
  sav2=sav2-ps1
  con2=jbot-(L(nit)+fracL(nit))
  call psi(con2,ps2)
  sav1=ps2-sav1
  t5=(0.0,1.0)*(sav2+endpoint*sav1)/tpp

  ip1=ip
  if (ip1.gt.(0.5*L(nit))) then
     ip1=max1(0.2*L(nit),1.0)
  endif
  do i=jbot,ip1

     !     Compute the value of the saddle point c. This has an approximate value
     !
     !      (i-phi1)/(2*phi2)-Sum[n=3 to nmax] n*((i-phi1)^(n-1))*phin/(2*phi2)^n +smaller terms
     !
     !     Unfortunately for large i the smaller terms, whilst small are not negligible. And they become
     !     inceasingly complex. It is possible to show that whilst in theory nmax=infinity one can cut if
     !     at a value nmax<=(mc+1-q)/l+2 at which point any higher order terms give a negligible contribution
     !     to the value of g(c).
     !     However, in turns out that as the length of the sum reduces g(c) becomes more complex and
     !     the value of g(c appropriate for I=L(nit)) goes from third order, to 4th order, to fifth order etc.
     !     This is determined by the paramter small found in the routine scheme. So as the scheme complexity increases
     !     so does the complexity associated with c.
     !     Rather than try and take this explcitly into account (as HAS to be done in routine scheme), here it is sufficient to
     !     compute c numerically to some specified tolerance. So for schemes beyond m(nit)>5 a Newton-Raphson method is
     !     used to compute c as in computenewton. So first a guess for c is computed. For schemes m(nit)=3,4 or 5 this is sufficiently accurate.
     !     For higher order schemes the guess is refined using Newton Raphson.

     con1=i*1.0-phicoeff(1,nit)
     wm=3*phicoeff(3,nit)*(con1**2)/(xr(nit)**3)
     if (m1(nit).ge.4) then
        wm=wm+(4*phicoeff(4,nit)-18*(phicoeff(3,nit)**2)/xr(nit))*(con1**3)/(xr(nit)**4)
     endif
     if (m1(nit).ge.5) then
        wm=wm+(5*(xr(nit)**2)*phicoeff(5,nit)+135*(phicoeff(3,nit)**3)-60*xr(nit)*phicoeff(3,nit) &
             *phicoeff(4,nit))*(con1**4)/(xr(nit)**7)
     endif
     c=con1/xr(nit)-wm
     if (m1(nit).gt.5) then
        call computecnewton(m1(nit),nit,con1,tol,c)
     endif
     sav1=sp*sx*c*sqrt(2.0)
     sav1=erfc(sav1)
     gc=i*c
     do j=1,m1(nit)
        gc=gc-phicoeff(j,nit)*(c**j)
     enddo
     cr1=(0.0,1.0)*tpp*gc
     cr1=exp(cr1)
     sav2=exp(tpm*con1*c)
     cr2=(0.0,1.0)*phicoeff(2,nit)/p
     !        cr2=(0.0,0.0)
     cr3=(1/con1+cr2*(1+tpp*c*con1*(1.0+p*con1*c))/(con1**3))
     sum0=sum0+(-cr1*sav1/(2*sx*EPI4)-(0.0,1.0)*(sav2*cr3-cr2/(con1**3))/tpp)
  enddo


  !      These are the ip terms for k=possibly 0,1,2,..ip for the lower endpoint. Basically there are two contributions
  !      sum and sum2. The integral is split into I4+I5+I6. I6=0, I5 contains the saddle. Sum is the I4 terms.
  !      sum2 gives the second order terms in I5 (the main terms being the saddles used in the next Gaussian sum
  !      iteration). The terms making up sum and sum2 are comparable in magnitude. The main saddle terms are
  !      considerably larger.

  sum1=(0.0,0.0)
  c1=L(nit)+fracL(nit)

  !      This ecor correction may be significant when muliplied by (i-phi1)**4 for large i=L(nit)

  !       ecor=9*p*(phicoeff(3,nit)**2)/(32.0*(phicoeff(2,nit)**5))
  ecor=0.0

  do i=L(nit),L(nit)-ip1+1,-1
     con1=i*1.0-phicoeff(1,nit)
     wm=3*phicoeff(3,nit)*(con1**2)/(xr(nit)**3)
     if (m1(nit).ge.4) then
        wm=wm+(4*phicoeff(4,nit)-18*(phicoeff(3,nit)**2)/xr(nit))*(con1**3)/(xr(nit)**4)
     endif
     if (m1(nit).ge.5) then
        wm=wm+(5*(xr(nit)**2)*phicoeff(5,nit)+135*(phicoeff(3,nit)**3)-60*xr(nit)*phicoeff(3,nit) &
             *phicoeff(4,nit))*(con1**4)/(xr(nit)**7)
     endif
     c=con1/xr(nit)-wm
     if (m1(nit).gt.5) then
        call computecnewton(m1(nit),nit,con1,tol,c)
     endif
     beta=c/L(nit-1)
     xbeta=L(nit-1)*(1-beta)
     sav1=sp*sx*xbeta*sqrt(2.0)
     sav1=erfc(sav1)
     gc=i*c
     do j=1,m1(nit)
        gc=gc-phicoeff(j,nit)*(c**j)
     enddo
     cr1=(0.0,1.0)*tpp*gc
     cr1=exp(cr1)
     cr2=(0.0,1.0)*phicoeff(2,nit)/p
     !        cr2=(0.0,0.0)
     sav2=exp(tpm*xbeta*(c1-i))
     cr3=(1/(c1-i)+cr2*(1+tpp*xbeta*(c1-i)*(1.0+p*(c1-i)*xbeta))/((c1-i)**3))
     sum1=sum1+(-sav1)*cr1/(2*sx*EPI4)
     sum1=sum1-((0.0,1.0)*endpoint*(sav2*cr3-cr2/((c1-i)**3))/tpp)

     !        if (i.eq.(L(nit)-ip1+1)) then
     !         sb(3)=sum1
     !        endif
     !        if (i.eq.(L(nit)-ip1+2)) then
     !         sb(2)=sum1
     !        endif
     !        if (i.eq.(L(nit)-ip1+3)) then
     !         sb(1)=sum1
     !        endif
  enddo

  t5=t5+(SUM0+SUM1)

  !       write(6,*) 'revised t5= ',t5

  !      These are the ip terms for k=L(nit),L(nit)-1 for the upper endpoint. Basically there are two contributions
  !      sum1 and sum3. The integral is split into I1+I2+I3. I1=0, I2 contains the saddle. Sum1 is the I3 terms.
  !      sum3 gives the second order terms in I2 (the main terms being the saddles used in the next Gaussian sum
  !      iteration). The terms making up sum1 and sum3 are comparable in magnitude. The main saddle terms are
  !      considerably larger. Note the exp(I*ecor) correction. One expect this to be very to one but for these large
  !      k values it about exp(I*0.05) which means it is worth including. All these terms make up t5.

  !       if(istart.gt.-1) then
  !        WRITE(6,1104) 'T1 VALUE= ',real(t1),aimag(t1),'*I  T2 VALUE= ',real(t2),aimag(t2),'*I'
  !        WRITE(6,1104) 'T3 VALUE= ',real(t3),aimag(t3),'*I  T4 VALUE= ',real(t4),aimag(t4),'*I'
  !        WRITE(6,1105) 'T5 VALUE= ',real(t5),aimag(t5),'*I'
  !       endif

  qq=conjg(t1+t2+t4)+t3+t5

1104 FORMAT(A10,E13.6,1X,E13.6,A14,E13.6,1X,E13.6,A2)
1105 FORMAT(A10,E13.6,1X,E13.6,A2)

END subroutine q




       SUBROUTINE gauleg(x1,x2,ab,we)

       implicit none

       integer, parameter :: dp = selected_real_kind(33)
       integer, parameter :: dp1 = selected_int_kind(16)
       
       real(dp) :: x1,x2,ab(64),we(64),eps,p1,p2,p3,pp,xl,xm,z,z1,p,sp,tpp,tpm
       integer :: i,j,m,n
       complex(kind=16) :: EPI4

       COMMON/PARS4/ P,SP,TPP,TPM,EPI4

       eps=3.0E-14
       n=64

       m=(n+1)/2
       xm=0.5*(x2+x1)
       xl=0.5*(x2-x1)

       do i=1,m
        z=cos(p*(i-0.25)/(n+0.5))

756     continue

        p1=1.0
        p2=0.0
        do j=1,n
         p3=p2
         p2=p1
         p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/(j*1.0)
        enddo

        pp=n*(z*p1-p2)/(z*z-1.0)
        z1=z
        z=z1-p1/pp
        if (abs(z-z1).gt.eps) goto 756
        ab(i)=xm-xl*z
        ab(n+1-i)=xm+xl*z
        we(i)=2*xl/((1.0-z*z)*pp*pp)
        we(n+1-i)=we(i)
       enddo
       return
       end subroutine gauleg


subroutine pari_calc(rzsum,iteration,numbercalc,rsphasear,N,NIS)





  use ISO_C_BINDING, only : C_PTR, C_DOUBLE
  use PARI
  implicit none


  integer, parameter   :: dp = selected_real_kind(33)
  integer(kind=C_LONG) :: prec   = 5 ! 152 decimal digits

  type(C_PTR)          :: t, rsphase_pari
  type(C_PTR)          :: u,v,w,SS

  real(dp), INTENT(OUT):: rzsum(15)
  real(dp), INTENT(IN) :: rsphasear(15)

  integer(kind=C_LONG) :: I
  integer(kind=C_LONG) :: av
  integer*8 :: I_START, I_END,iteration,numbercalc,N,NIS,j,k



  CALL pari_init(10000000_8, 2_8)


  av = get_avma()
  
  SS = stor(1_8,prec)
  SS = gdiv(SS, stor(10_8,prec) )
  SS = gdiv(SS, stor(10_8,prec) )
  
  do j=1,2*numbercalc-1
   CALL t_grabber(t)
   if (j.lt.numbercalc) then
      do k=1,(numbercalc-j)
         t = gsub(t,SS)
      enddo
   endif
   if (j.gt.numbercalc) then
    do k=1,(j-numbercalc)
     t = gadd(t,SS)
    enddo
   endif
   
   CALL real2pari(rsphasear(j),rsphase_pari)

!  iteration = 2
  !  N = 100000

  if (NIS.eq.0) then
     I_START = 1 + (iteration-1) * N
  else
     I_START = NIS
  endif   
  I_END   =         iteration * N

  w = stor(0_8,prec)

  DO I=I_START,I_END



     u =  gmul(t, glog( stor(I,prec) ,prec )  )  
     u =  gsub(u , rsphase_pari )
     u =  gcos( u, prec         )
     v =  gsqrt( stor(I,prec) , prec)
     u =  gdiv(u,v)
     w =  gadd( u , w )


     if( modulo(I,1000) .EQ. 0) then


        ! This is a 'clear the stack' except w.
        !
        w = gerepilecopy(av,w)
        SS = stor(1_8,prec)
        SS = gdiv(SS, stor(10_8,prec) )
        SS = gdiv(SS, stor(10_8,prec) )
        CALL t_grabber(t)
        if (j.lt.numbercalc) then
         do k=1,(numbercalc-j)
          t = gsub(t,SS)
         enddo
        endif
        if (j.gt.numbercalc) then
         do k=1,(j-numbercalc)
          t = gadd(t,SS)
         enddo
        endif
        CALL real2pari(rsphasear(j),rsphase_pari)

     endif

  ENDDO

  rzsum(j) = rzsum(j) + rtodbl(w)

!  print*, 'rzsum = ', rzsum

  !       tlnXmod2pi = rtodbl(  gmod( gmul( t , glngamma( stoi( b2  ) ,prec ) )   , Pi2n(1_8, prec))    )

  enddo

  CALL pari_close()

END subroutine pari_calc




subroutine t_grabber(t)


  use ISO_C_BINDING, only : C_PTR, C_DOUBLE
  use PARI
  implicit none

  integer(kind=C_LONG) :: prec   = 5 ! 152 decimal digits

  type(C_PTR)            :: u1, u2, u3, u4, u5, u6
  type(C_PTR)            :: t

  integer(kind=C_LONG)   ::  c1, c2, c3, c4, c5, n 
  CHARACTER(LEN=45)      ::   Y

  COMMON/PARS9/ Y

  !       CALL pari_init(10000000_8, 2_8)

!  Y = '1.0000000000000000000000056100000000000000e24'  ! keep decimal digits 40 long
  ! Y = '3.0123456789987654321000000000000000000000e32'  ! example input.


  READ(Y(1:1)  , '(I1)'  )  c1            ! first digit and 10-digit pieces of t
  READ(Y(3:12)  , '(I10)'  )  c2
  READ(Y(13:22)  , '(I10)'  )  c3
  READ(Y(23:32)  , '(I10)'  )  c4
  READ(Y(33:42)  , '(I10)'  )  c5
  READ(Y(44:45)  , '(I10)'  )  n


  !      READ(X, '(I19)') b3


  u1 = gpow( stoi(10_8), stoi(  0_8 ), prec)    ! construct t in blocks of 10 digits
  u2 = gpow( stoi(10_8), stoi(  -10_8 ), prec)
  u3 = gpow( stoi(10_8), stoi(  -20_8 ), prec)
  u4 = gpow( stoi(10_8), stoi(  -30_8 ), prec)
  u5 = gpow( stoi(10_8), stoi(  -40_8 ), prec)
  u6 = gpow( stoi(10_8), stoi(  n ), prec)


  t = gadd( gadd( gadd( gadd(                           &
       gmul(stor(c1,prec) , u1)        ,           &
       gmul(stor(c2,prec) , u2)      ) ,           &
       gmul(stor(c3,prec) , u3)      ) ,           &
       gmul(stor(c4,prec) , u4)      ) ,           &
       gmul(stor(c5,prec) , u5)      )
  !  constructed high precision t value.

  t = gmul(t,u6)



  !        CALL pari_close()

!  PRINT*, 't = ', rtodbl(t)



END subroutine t_grabber




subroutine real2pari(r,r_pari)


  use ISO_C_BINDING, only : C_PTR, C_DOUBLE
  use PARI
  implicit none

  integer, parameter   :: dp = selected_real_kind(33)
  integer(kind=C_LONG) :: prec   = 5 ! 152 decimal digits

  type(C_PTR)            :: u0, u1, u2, u3, u4, u5, u6
  type(C_PTR)            :: r_pari

  integer(kind=C_LONG)   :: c0, c1, c2, c3, c4, c5, n
  CHARACTER(LEN=70)      ::   Y

  real(dp)               ::  r

  !       CALL pari_init(10000000_8, 2_8)

!  r = 0.029471234


     WRITE(Y, '(F68.45)') r

!    PRINT*, 'Y = ', Y

!    Y = ' 0.5811585840209867046077257957962776568491'
!    Y = ' 0.581158584020986704607720000000000049'


!  Y = '3.1234567876543456'

    READ(Y(3:12)   , '(I10)'  )  c0
    READ(Y(13:22)  , '(I10)'  )  c1            ! first digit and 10-digit pieces of r
    READ(Y(24:33)  , '(I10)'  )  c2
    READ(Y(34:43)  , '(I10)'  )  c3
    READ(Y(44:53)  , '(I10)'  )  c4
    READ(Y(54:63)  , '(I10)'  )  c5

!    PRINT*, 'c1 = ', c0
  !      READ(X, '(I19)') b3


  u0 = gpow( stoi(10_8), stoi(   10_8 ), prec)
  u1 = gpow( stoi(10_8), stoi(    0_8 ), prec)    ! construct t in blocks of 10 digits
  u2 = gpow( stoi(10_8), stoi(  -10_8 ), prec)
  u3 = gpow( stoi(10_8), stoi(  -20_8 ), prec)
  u4 = gpow( stoi(10_8), stoi(  -30_8 ), prec)
  u5 = gpow( stoi(10_8), stoi(  -40_8 ), prec)


    r_pari = gadd( gadd( gadd( gadd( gadd(         &
       gmul(stor(c0,prec) , u0)        ,           &
       gmul(stor(c1,prec) , u1)      ) ,           &
       gmul(stor(c2,prec) , u2)      ) ,           &
       gmul(stor(c3,prec) , u3)      ) ,           &
       gmul(stor(c4,prec) , u4)      ) ,           &
       gmul(stor(c5,prec) , u5)      )
  !  constructed high precision t value.




  !        CALL pari_close()




END subroutine real2pari





subroutine pari_erfc(z,rz,iz)





  use ISO_C_BINDING, only : C_PTR, C_DOUBLE
  use PARI
  implicit none


  integer, parameter   :: dp = selected_real_kind(33)
  integer(kind=C_LONG) :: prec   = 5 ! 152 decimal digits

  type(C_PTR)          :: z_pari
  type(C_PTR)          :: u, S

  real(dp)             :: z, rz, iz



  CALL pari_init(10000000_8, 2_8)

  CALL real2pari(z,z_pari)   ! I THINK THIS IS A PLACE AN ERROR MIGHT POP UP


  S = gatan( stor(1_8 , prec) ,prec)        ! PI/4
  u = gsqrt( stoi(-1_8) ,prec)              ! imag(1)
  S = gmul ( S, u      )                     ! I*pi/4
  S = gmul ( S, z_pari )                     ! Z*I*pi/4
  S = gexp ( S, prec   )                     ! exp( Z*I*PI/4)
  S = gerfc( S, prec   )                     ! erfc( above )

  rz = rtodbl( greal(S) )
  iz = rtodbl( gimag(S) )

!  print*, 'imag(S) = ', rz
!  print*, 'real(S) = ', iz



  CALL pari_close()

END subroutine pari_erfc





subroutine pari_tse(rae,numbercalc,tse)

        ! Routine to calculate:   tse = (t*SE) mod 2pi

  use ISO_C_BINDING, only : C_PTR, C_DOUBLE
  use PARI
  implicit none


  integer, parameter   :: dp = selected_real_kind(33)
  integer, parameter   :: dp1 = selected_int_kind(16)
  integer(kind=C_LONG) :: prec   = 5 ! 152 decimal digits change 5
  integer(kind=C_LONG) :: av         ! change 6

  type(C_PTR)          :: t, rae_pari, SE_pari
  type(C_PTR)          :: u, S, e, pc, v, SS

  real(dp), INTENT(IN) :: rae
  real(dp), INTENT(OUT):: tse(15)
  real(dp)             :: rae1
  real                 :: frac_part
  integer(dp1)         :: i,j 
  integer*8            :: numbercalc
  
!  rae = 2.5

  !  CALL pari_init(10000000_8, 2_8) change 3
  
  av = get_avma()    !change 7
  
  CALL real2pari(rae,rae_pari)   ! I THINK THIS IS A PLACE AN ERROR MIGHT POP UP

  SS = stor(1_8,prec)
  SS = gdiv(SS, stor(10_8,prec) )
  SS = gdiv(SS, stor(10_8,prec) )
  
  do i=1,2*numbercalc-1
     CALL t_grabber(t)
     
   if (i.lt.numbercalc) then
      do j=1,(numbercalc-i)
         t = gsub(t,SS)
      enddo
   endif
   if (i.gt.numbercalc) then
      do j=1,(i-numbercalc)
         t = gadd(t,SS)
      enddo
   endif   

   
  frac_part = rtodbl( gfrac(rae_pari) )

  ! Manual rounding of rae

  if(frac_part.ge.0.5) then

          rae1 = ceiling(rae,dp)

  else

          rae1 = floor(rae,dp)

  endif


  CALL real2pari(rae1,rae_pari)


  S = stor(8_8,prec)
  u = Pi2n(1_8, prec)
  u = gdiv(u, stor(2_8,prec) )
  S = gmul(t,S)
  S = gdiv(u,S)

 ! print*, 'S = ', rtodbl(S)

  S = gsqrt(S,prec)    ! We are up to S = 1/ sqrt( t*8 / pi ) - which is ain

  e = gmul(rae_pari, S) ! this is 'e'


  u = gmul(e,e)
  u = gmul( stor(2_8,prec) , u )

  v = gmul(e,e)
  v = gdiv( stor(1_8,prec), v )
  v = gsub(stor(1_8,prec),v)
  v = gsqrt(v,prec)
  v = gadd( stor(1_8,prec) , v )

  S = gmul(u,v)
  pc = gsub(S , stor(1_8,prec) )


  u = glog(pc,prec)


  v = gdiv( stor(1_8,prec), pc)
  S = gadd( u , v )
  S = gadd( S , stor(1_8,prec) )
  SE_pari = gdiv( S , stor(2_8,prec) )

!  PRINT*, 'SE_pari = ', rtodbl(SE_pari)

  S = gmul(t,SE_pari)
  S = gmod(S, Pi2n(1_8, prec))

  tse(i) = rtodbl(S)

  enddo


  CALL set_avma(av)   ! change 8

!  CALL pari_close() change 4

END subroutine pari_tse


subroutine pari_phases(yphase,rsphase,a)

        ! Routine to calculate:   yphase     = (t+pi/8) mod 2pi
        !                         theta(t)   = IM( log(Gamma(1/4+i*t/2))  -t*log(pi)/2 )
        !                         rsphase(t) = theta(t) mod 2pi

  use ISO_C_BINDING, only : C_PTR, C_DOUBLE
  use PARI
  implicit none



  integer, parameter   :: dp = selected_real_kind(33)
  integer(kind=C_LONG) :: prec   = 25 ! 152 decimal digits
  integer(kind=C_LONG) :: av


  type(C_PTR)          :: t,u,v,w,x,y
  type(C_PTR)          :: yphase_pari, theta_pari, rsphase_pari, a_pari

  real(dp), INTENT(OUT):: yphase, rsphase, a

  yphase = 0.0
  rsphase= 0.0


  CALL pari_init(10000000_8, 2_8)


    CALL t_grabber(t)


    v           = Pi2n(1_8, prec) ! 2pi
    w           = gsqrt( stoi(-1_8) ,prec) ! imag(1)
    y           = gdiv(v, stor(2_8,prec)) ! pi
    u           = gdiv(y, stor(8_8,prec)) ! pi/8
    u           = gadd(t,u)               ! t + pi/8
    yphase_pari = gmod(u, v)              ! t + pi/8 mod 2pi

    CALL rtoquad(yphase_pari,yphase)
!    yphase      = rtodbl(yphase_pari)
    
    u           = gdiv(t, stor(2_8,prec))                       ! t/2
    u           = gmul(w, u)                                    ! i*t/2
    x           = gdiv( stor(1_8,prec), stor(4_8,prec) )        ! 1/4
    u           = gadd(u,x)                                     ! 1/4 + i*t/2
    u           = glngamma(u,prec)                              ! log(gamma( 1/4 + i*t/2 ))
    u           = gimag( u )                                    ! imag( above )

    x           = glog(y,prec)                                  ! log(pi)
    x           = gdiv(x, stor(-2_8,prec) )                     ! -log(pi)/2
    x           = gmul(x,t)                                     ! -t*log(pi)/2
    theta_pari  = gadd(u,x)                                     ! imag(log(gamma( 1/4 + i*t/2 ))) - t*log(pi)/2
    rsphase_pari = gmod(theta_pari, v)                          ! rsphase = (above) mod 2pi

!    rsphase       = rtodbl(rsphase_pari)
    CALL rtoquad(rsphase_pari,rsphase)

    u           = gdiv(t, y)                                    ! t/pi
    v           = gmul(u, stor(8_8,prec))                       ! 8*t/pi
    a_pari      = gsqrt(v ,prec)                                ! sqrt(8*t/pi)

    CALL rtoquad(a_pari,a)
    
  CALL pari_close()

END subroutine pari_phases

subroutine rtoquad(pari_input,var)


  use ISO_C_BINDING, only : C_PTR, C_DOUBLE
  use PARI
  implicit none



  integer, parameter   :: dp = selected_real_kind(33)
  integer, parameter   :: dp1 = selected_int_kind(16)
  integer(kind=C_LONG) :: prec   = 25 ! 152 decimal digits
  integer(kind=C_LONG) :: av, m

  type(C_PTR)          :: n,x,y,y1,y2,pari_input
  type(C_PTR)          :: expo, scale_d, scale_u

  real(dp)             :: var, var1
  real*8               :: var1_dbl, var2_dbl

  integer(dp1)         :: num 

  num = 10000000
 
!  CALL pari_init(10000000_8, 2_8)



     y           = pari_input
!    y           = Pi2n(1_8, prec) ! 2pi    
!    y           = gdiv(y, stor(2_8,prec)) ! pi pari
!    y           = gmul(y, stor(1_8,prec)) ! 100*pi


    m           = sizedigit(  y  ) -1

    if(m .le. 0) then
      m =  -sizedigit( gdiv( stoi(1_8), y  ))
    endif


    n           =  stoi( m )     !gsub(sizedigit(y), stoi(1_8) ) ! n-1    (for 1000, n = 4 as there are 4 digits)

 

    scale_d     = gpow( stoi(10_8), gmul( stor(-1_8,prec), n) , prec)
    scale_u     = gpow( stoi(10_8), n, prec)
    expo        = gsqr( stor(10000000_8, prec) ) ! 10^14




    y           = gmul(y,scale_d)

    y2          = gmul(y, expo )  !expo*pi
    x           = gfloor(y2)      ! integer part (expo * pi)
    y2          = gsub(y2, x)     ! decimal part (expo * pi)

    y1          = gdiv(x,  expo ) ! first part of pi
    y2          = gdiv(y2, expo ) ! second part of pi

    var   = 0.0
    var1  = 0.0

    var = rtodbl(y1)
    var = var * num
    var = var * num
    
!    print*,'var = ',var
    var = nint(var,dp1)
!    print*,'floor var = ',var
    var1= rtodbl(y2)
    var1= var1* num
    var1= var1* num
!    print*,'var1 = ',var1
    var = var + var1
    var = var / num
    var = var / num



    if(m>0) then    
    var = var *  10**m
    else   
    var = var / 10**(abs(m))
    endif

 END subroutine rtoquad 
