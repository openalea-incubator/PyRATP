module RATP

 use constant_values
 use grid3D
 use skyvault
 use vegetation_types
 use dir_interception
 use hemi_interception
 use shortwave_balance
 use micrometeo
 use energy_balance
 use Photosynthesis
 use MinerPheno

 integer :: numx_out(100), numy_out(100), numz_out(100), kxyz_out(100)
 integer ::  form_vgx=55
 integer :: fileTypeArchi = 1
 integer :: val = 1
 integer :: iterspatial = 0
 integer :: itertree = 0
 character*200 fname

 character(len=17):: pathResult

 integer :: nbvoxelveg = 1
 integer :: nbiter = 0

 !character*2 hhx, hhy, hhz

 real, allocatable :: out_time_spatial(:,:), out_time_tree(:,:),out_rayt(:,:)

contains
!----------------------------

 subroutine do_all_mine


 !write(*,*)
 write(*,*)  ' R. A. T. P. Mineuse   Version 2.0'
 !write(*,*)  ' Radiation Absorption, Transpiration and Photosynthesis'
 !write(*,*)
 !write(*,*)  ' Spatial distribution in a 3D grid of voxels'
 !write(*,*)
 !write(*,*)  '                July 2003'
 !write(*,*)

 !write(*,*)
 !write(*,*)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! spec_grid='grd'     ! definition de la grille
! spec_vegetation='veg'   ! definition des types de végétation

! spec_gfill='dgi'     ! definition du fichier de structure (feuillage)
!spec_mmeteo='mto'     ! definition du fichier mmeteo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 call cv_set

 dpx=dx/5.
 dpy=dy/5.

 if (int_scattering.eq.1)  then
  scattering=.TRUE.
 else
  scattering=.FALSE.
 end if
 !write(*,*) "scattering", scattering
 
 if (int_isolated_box.eq.1)  then
  isolated_box=.TRUE.
 else
  isolated_box=.FALSE.
 end if
 !write(*,*) "isolated", isolated_box
 call Farquhar_parameters_set
 !write(*,*) 'doall'

 call hi_doall(dpx,dpy,isolated_box)  ! Compute interception of diffuse and scattering radiation, ie exchange coefficients
 !ttot
 !pathResult = 'c:/tmpRATP/Resul/'
 pathResult = '/tmp/tmpRATP/Resul/'
 fname=pathResult//'output_PARclasses.dat'
 !open (2,file=fname)
 !write(2,*) 'ntime day hour vt PARg %SF50 %SF100'

 fname=pathResult//'output_tsclasses.dat'
 !open (3,file=fname)
 !write(3,*) 'ntime day hour vt Tair %SF11 %SF12'

 fname=pathResult//'output_minerpheno.dat'
 !open (10,file=fname)
 !write(10,*) 'ntime day hour sumTair Nlarvaout Nlarvadead Tair'

! Leaf temperature at the voxel scale

 fname = pathResult//'output_leafTemp.dat'
 !open (12,file=fname)
 !write(12,*) 'ntime day hour voxel Tsh Tshm Tsl Tslm Tair Tbody(10xx mort,+50xx out)'


! Memory allocation in MinerPheno module
 !write(*,*) 'miph_destroy'
 call miph_destroy
 !write(*,*) 'miph_allocate'
 call miph_allocate ! includes Variable initiation to 0

 ntime=0
 endmeteo=.FALSE.
 call mm_initiate

 itertree = 0
 iterspatial = 0
 do while (.NOT.((endmeteo).OR.((nlarvaout+nlarvadead).ge.voxel_canopy(2))))
  ntime=ntime+1
 ! write(*,*) '...Iteration : ',ntime,nbli
  call mm_read(ntime,nbli)  ! Read micrometeo data (line #ntime in file mmeteo.<spec>)
  !write(*,*) '...mm_read : '
  call swrb_doall     ! Compute short wave radiation balance
  !write(*,*) '...swrb_doall : '

  call eb_doall_mine    ! Compute energy balance
  !write(*,*) '...eb_doall_mine : '
  call miph_doall     ! Compute miner larva development
  !write(*,*) '...miph_doall : '
  !write(*,*) 'nent ',nent

  do jent=1,nent
   itertree = itertree +1
   out_time_tree(itertree,1) = ntime
   out_time_tree(itertree,2) = day
   out_time_tree(itertree,3) = hour
   out_time_tree(itertree,4) = jent
   out_time_tree(itertree,5) = glob(ntime)*2.02/0.48
   do class = 1,45
    out_time_tree(itertree,class+5) = Spar(jent,class)/S_vt(jent)
   end do
   out_time_tree(itertree,51) = taref
   do class = 1,45
    out_time_tree(itertree,class+51) = Sts(jent,class)/S_vt(jent)
   end do
   out_time_tree(itertree,97) = A_canopy
   out_time_tree(itertree,98) = E_canopy
   !write(2,20) ntime, day, hour, jent, glob(1)*2.02/0.48, (Spar(jent,class)/S_vt(jent), class=1,45) ! %SF per irradiance class
   !write(3,30) ntime, day, hour, jent, taref, (Sts(jent,class)/S_vt(jent), class=10,45)
  end do

  !write(10,70) ntime, day, hour, sum_taref, Nlarvaout, Nlarvadead, taref !, sum_dev_rate(1), sum_dev_rate(10), sum_dev_rate(100)

  !if (hour.eq.12) then
  do k=1,nveg
   do je=1,nje(k)
     iterspatial = iterspatial +1
     jent=nume(je,k)
     if (ismine(jent).eq.1) then
      out_time_spatial(iterspatial,1) = ntime
       out_time_spatial(iterspatial,2) = day
       out_time_spatial(iterspatial,3) = hour
       out_time_spatial(iterspatial,4) = taref
       out_time_spatial(iterspatial,5) = k
    !Boucler sur nombre vegetation
       out_time_spatial(iterspatial,6) = ts(0,je,k)
       out_time_spatial(iterspatial,7) = ts(1,je,k)
    !Boucler sur nombre vegetation
       out_time_spatial(iterspatial,8) = A_detailed(0,je,k)
       out_time_spatial(iterspatial,9) = A_detailed(1,je,k)
       out_time_spatial(iterspatial,10) = E(0,je,k)
       out_time_spatial(iterspatial,11) = E(1,je,k)
       out_time_spatial(iterspatial,12) = S_detailed(0,je,k)
       out_time_spatial(iterspatial,13) = S_detailed(1,je,k) 
       out_time_spatial(iterspatial,14) = gs(0,je,k)
       out_time_spatial(iterspatial,15) = gs(1,je,k)

!      if (larvadeath(k).gt.0) then
       !write(12,90) ntime, day, hour, k, ts(0,1,k), ts(0,2,k), ts(1,1,k), ts(1,2,k), taref, 1000 + ntime + .0
!       out_time_spatial(iterspatial,14) = 1000 + ntime + .0
!      else if  (larvaout(k).gt.0) then
!       write(12,90) ntime, day, hour, k, ts(0,1,k), ts(0,2,k), ts(1,1,k), ts(1,2,k), taref, 5000 + ntime + .0
!       out_time_spatial(iterspatial,14) = 5000 + ntime + .0
!      else
!       write(12,90) ntime, day, hour, k, ts(0,1,k), ts(0,2,k), ts(1,1,k), ts(1,2,k), taref, tbody(k)
!       out_time_spatial(iterspatial,14) = tbody(k)
!      end if
     end if
   end do
  end do
  !end if

  if (ntime.eq.nbli) then
    endmeteo=.TRUE.
  end if

 end do

 nbiter =nbiter + ntime


! close (1)
! close (2)
! close (3)
! close (10)
! close (12)

! Mine phenology at voxel scale, at the end of the simulation period

 fname = pathResult//'output_minerspatial.dat'
 !open (11,file=fname)
 !write(11,*) 'Voxel# jx jy jz time_death time_out sum_tair_out'




! do k=1,nveg
!  if (sum_dev_rate(k).gt.0.) then   ! Voxel k includes a miner
!   write(11,80) k, numx(k), numy(k), numz(k), larvadeath(k), larvaout(k), sum_tair(k)
!  endif
! end do

! close (11)



 !pause

10 format(i4,1x,f4.0,1x,f5.2,2(1x,f5.3),2(1x,f7.3))
11 format(i4,1x,f4.0,1x,f5.2,12(1x,f7.3))
12 format(i4,1x,f4.0,1x,f6.3,10(1x,i2,1x,2(f6.2,1x),2(f6.0,1x),2(f9.6,1x)))
20 format(i4,1x,f4.0,1x,f6.2,1x,i2,1x,f5.0,50(1x,f8.6))
30 format(i4,1x,f4.0,1x,f6.2,1x,i2,1x,f6.2,50(1x,f5.3))
70 format(i4,1x,f4.0,1x,f6.2,1x,f9.0,2(1x,i5),1x,f6.2,3(1x,f6.4))
80 format(i5,1x,3(i3,1x),2(i4,1x),f8.1)
90 format(i4,1x,f4.0,1x,f6.2,1x,i5,6(1x,f8.3))


! Deallocation des tableaux

 !call g3d_destroy
 !call sv_destroy
 !call vt_destroy
 !call mm_destroy
 call di_destroy
 call hi_destroy
 call swrb_destroy
 call eb_destroy
 call ps_destroy
 call miph_destroy

 !pause
 !z = sin(x+y)
 write(*,*) 'CALCULS TERMINES 1'
 end subroutine do_all_mine



subroutine do_all

 !write(*,*)
 write(*,*)  ' R. A. T. P.    Version 2.0'
 !write(*,*)  ' Radiation Absorption, Transpiration and Photosynthesis'
 !write(*,*)
 !write(*,*)  ' Spatial distribution in a 3D grid of voxels'
 !write(*,*)
 !write(*,*)  '                July 2003'
 !write(*,*)

 !write(*,*)
 !write(*,*)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! spec_grid='grd'     ! definition de la grille
! spec_vegetation='veg'   ! definition des types de végétation

! spec_gfill='dgi'     ! definition du fichier de structure (feuillage)
!spec_mmeteo='mto'     ! definition du fichier mmeteo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 call cv_set

 dpx=dx/5.
 dpy=dy/5.

 if (int_scattering.eq.1)  then
  scattering=.TRUE.
 else
  scattering=.FALSE.
 end if
 !write(*,*) "scattering", scattering
 
 if (int_isolated_box.eq.1)  then
  isolated_box=.TRUE.
 else
  isolated_box=.FALSE.
 end if
 !write(*,*) "isolated", isolated_box

 call Farquhar_parameters_set

 call hi_doall(dpx,dpy,isolated_box)  ! Compute interception of diffuse and scattering radiation, ie exchange coefficients
          
 !pathResult = 'c:/tmpRATP/Resul/'
 pathResult = '/tmp/tmpRATP/Resul/'
 fname=pathResult//'output_PARclasses.dat'
 !open (2,file=fname)
 !write(2,*) 'ntime day hour vt PARg %SF50 %SF100'

 fname=pathResult//'output_tsclasses.dat'
 !open (3,file=fname)
 !write(3,*) 'ntime day hour vt Tair %SF11 %SF12'

! Leaf temperature at the voxel scale
 fname = pathResult//'output_leafTemp.dat'
 !open (12,file=fname)
 !write(12,*) 'ntime day hour voxel Tsh Tsl Tair'


 ntime=0
 endmeteo=.FALSE.
 call mm_initiate
    
 itertree = 0
 iterspatial = 0
 do while (.NOT.((endmeteo)))
  ntime=ntime+1
  write(*,*) '...Iteration : ',ntime,nbli
  call mm_read(ntime,nbli)  ! Read micrometeo data (line #ntime in file mmeteo.<spec>)
  !write(*,*) '...mm_read : '
  call swrb_doall     ! Compute short wave radiation balance

  call eb_doall
  call ps_doall
  !write(*,*) '...swrb_doall : '
  do jent=1,nent
   itertree = itertree +1
   out_time_tree(itertree,1) = ntime
   out_time_tree(itertree,2) = day
   out_time_tree(itertree,3) = hour
   out_time_tree(itertree,4) = jent
   out_time_tree(itertree,5) = glob(1)+glob(2)
   !do class = 1,45
    !out_time_tree(itertree,class+5) = Spar(jent,class)/S_vt(jent)
   !end do
   !out_time_tree(itertree,51) = taref
   !do class = 1,45
    !out_time_tree(itertree,class+51) = Sts(jent,class)/S_vt(jent)
   !end do
   !out_time_tree(itertree,97) = A_canopy
   !out_time_tree(itertree,98) = E_canopy
   out_time_tree(itertree,6) = taref
   out_time_tree(itertree,7) = A_canopy
   out_time_tree(itertree,8) = E_canopy
   
   !write(2,20) ntime, day, hour, jent, glob(1)*2.02/0.48, (Spar(jent,class)/S_vt(jent), class=1,45) ! %SF per irradiance class
   !write(3,30) ntime, day, hour, jent, taref, (Sts(jent,class)/S_vt(jent), class=10,45)
  end do


  !if (hour.eq.12) then
  do k=1,nveg
   do je=1,nje(k)
     iterspatial = iterspatial +1
     jent=nume(je,k)
       !write(*,*) k,je,jent
       out_time_spatial(iterspatial,1) = ntime
       out_time_spatial(iterspatial,2) = day
       out_time_spatial(iterspatial,3) = hour
       out_time_spatial(iterspatial,4) = taref
       out_time_spatial(iterspatial,5) = k
    !Boucler sur nombre vegetation
       out_time_spatial(iterspatial,6) = ts(0,je,k)
       out_time_spatial(iterspatial,7) = ts(1,je,k)
    !Boucler sur nombre vegetation
       out_time_spatial(iterspatial,8) = A_detailed(0,je,k)
       out_time_spatial(iterspatial,9) = A_detailed(1,je,k)
       out_time_spatial(iterspatial,10) = E(0,je,k)
       out_time_spatial(iterspatial,11) = E(1,je,k)
       out_time_spatial(iterspatial,12) = S_detailed(0,je,k)
       out_time_spatial(iterspatial,13) = S_detailed(1,je,k) 
       out_time_spatial(iterspatial,14) = gs(0,je,k)
       out_time_spatial(iterspatial,15) = gs(1,je,k)
       out_time_spatial(iterspatial,16) = N_detailed(je,k)      !ajout N foliaire, ngao 05/06/2013
       
       !write(12,90) ntime, day, hour, k, ts(0,1,k), ts(1,1,k), taref
   end do
  end do
  !end if

  if (ntime.eq.nbli) then
    endmeteo=.TRUE.
  end if

 end do

 nbiter =nbiter + ntime


! close (1)
! close (2)
! close (3)
! close (12)




10 format(i4,1x,f4.0,1x,f5.2,2(1x,f5.3),2(1x,f7.3))
11 format(i4,1x,f4.0,1x,f5.2,12(1x,f7.3))
12 format(i4,1x,f4.0,1x,f6.3,10(1x,i2,1x,2(f6.2,1x),2(f6.0,1x),2(f9.6,1x)))
20 format(i4,1x,f4.0,1x,f6.2,1x,i2,1x,f5.0,50(1x,f8.6))
30 format(i4,1x,f4.0,1x,f6.2,1x,i2,1x,f6.2,50(1x,f5.3))
70 format(i4,1x,f4.0,1x,f6.2,1x,f9.0,2(1x,i5),1x,f6.2,3(1x,f6.4))
80 format(i5,1x,3(i3,1x),2(i4,1x),f8.1)
90 format(i4,1x,f4.0,1x,f6.2,1x,i5,3(1x,f8.3))


! Deallocation des tableaux

! DEBUG the deallocation
 ! call g3d_destroy
 !call sv_destroy
 !call vt_destroy
 !call mm_destroy
 call di_destroy
 call hi_destroy
 call swrb_destroy
 call eb_destroy
 call ps_destroy

 write(*,*) 'CALCULS TERMINES 2'
 end subroutine do_all

 subroutine do_interception

 !write(*,*)
  write(*,*)  ' R. A. T. P.    Version 2.0'
 !write(*,*)  ' Radiation Absorption, Transpiration and Photosynthesis'
 !write(*,*)
 !write(*,*)  ' Spatial distribution in a 3D grid of voxels'
 !write(*,*)
 write(*,*)  '                July 2003 - December 2012  '
 !write(*,*)

 !write(*,*)
 !write(*,*)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! spec_grid='grd'     ! definition de la grille
! spec_vegetation='veg'   ! definition des types de végétation

! spec_gfill='dgi'     ! definition du fichier de structure (feuillage)
!spec_mmeteo='mto'     ! definition du fichier mmeteo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! call qui permet de tester l existance des tableaux dynamiques et de les vider s'ils existent
 call out_rayt_destroy
 call cv_set

 dpx=dx/5.
 dpy=dy/5.

 if (int_scattering.eq.1)  then
  scattering=.TRUE.
 else
  scattering=.FALSE.
 end if
 !write(*,*) "scattering", scattering
 
 if (int_isolated_box.eq.1)  then
  isolated_box=.TRUE.
 else
  isolated_box=.FALSE.
 end if
 !write(*,*) "isolated", isolated_box
 call hi_doall(dpx,dpy,isolated_box)  ! Compute interception of diffuse and scattering radiation, ie exchange coefficients
 ntime=0
 endmeteo=.FALSE.
 call mm_initiate

 allocate(out_rayt(nbli*nveg*nemax,9))
 iterspatial = 0
 do while (.NOT.((endmeteo)))
  ntime=ntime+1
  write(*,*) '...Iteration : ',ntime,nbli
  call mm_read(ntime,nbli)  ! Read micrometeo data (line #ntime in file mmeteo.<spec>)
  !write(*,*) '...mm_read : '
  call swrb_doall     ! Compute short wave radiation balance



  do k=1,nveg

   do je=1,nje(k)
     iterspatial = iterspatial +1
     jent=nume(je,k)

    ! Sortie rayonnement
       out_rayt(iterspatial,1) = jent
       out_rayt(iterspatial,2) = ntime
       out_rayt(iterspatial,3) = day
       out_rayt(iterspatial,4) = hour
       out_rayt(iterspatial,5) = k
    ! cf xintav
       out_rayt(iterspatial,6) = PARirrad(0,je,k)
       out_rayt(iterspatial,7) = PARirrad(1,je,k)
       out_rayt(iterspatial,8) = S_detailed(0,je,k)
       out_rayt(iterspatial,9) = S_detailed(1,je,k) 
 !      out_rayt(iterspatial,10) = N_detailed(je,k)

       !write(12,90) ntime, day, hour, k, ts(0,1,k), ts(1,1,k), taref
   end do
  end do
  !end if

  if (ntime.eq.nbli) then
    endmeteo=.TRUE.
  end if

 end do

 nbiter =nbiter + ntime

! Deallocation des tableaux

 !call sv_destroy
 !call vt_destroy
 call di_destroy
 call hi_destroy
 call swrb_destroy

 write(*,*) 'CALCULS TERMINES INTERCEPTION'
 end subroutine do_interception

 subroutine out_rayt_destroy
  !write(*,*) 'destroy out_rayt'
  if (allocated(out_rayt))  deallocate(out_rayt)
 end subroutine out_rayt_destroy


end module RATP

