#include <iostream>
#include <mpi.h>

#include "KernelAB_AOS_Base.hh"
#include "KernelAB_SOA_Base.hh"
#include "KernelAB_SOA_Pull.hh"
#include "KernelAB_SOA_Unroll.hh"

#include "KernelAB_SOA_Unroll_Pull.hh"

#include "PropagationAB.hh"

using namespace std;

/******************************************************************************/
void PropagationAB::setup(void)
{
   // initialize storage arrays
   commOffset_ = comm_.getCommLength();
   int dstrbSize = geometry_.getNumFluidPts() * _LATTICESIZE_;
#if defined(USE_KOKKOS)
   dstrb_d_= myViewPDF("dstrb_d_",dstrbSize);
   dstrb2_d_= myViewPDF("dstrb2_d_",dstrbSize);
   dstrb_h_ = Kokkos::create_mirror_view(dstrb_d_);
#else
#ifdef USE_SYCL
	dstrb_d_=sycl::malloc_device<Pdf>(dstrbSize,q_);
	dstrb2_d_=sycl::malloc_device<Pdf>(dstrbSize,q_);
#endif
   dstrb_ = new Pdf[dstrbSize];
   dstrb2_ = new Pdf[dstrbSize];
   fill(dstrb_, dstrb_+dstrbSize, 0.0);
   fill(dstrb2_, dstrb2_+dstrbSize, 0.0);
#endif

   // select kernel
   kernel_ = 0;
#ifdef SOA
   if (parameters_.getUnrollSetting() == 1 && _LATTICESIZE_ == 19)
   {
      if (parameters_.getPullSetting() == 1)
      {
#ifdef USE_SYCL
	kernel_ = new KernelAB_SOA_Unroll_Pull(lattice_, geometry_,q_);
#else
         kernel_ = new KernelAB_SOA_Unroll_Pull(lattice_, geometry_);
#endif
      }
      else
      {
#ifdef USE_SYCL
	kernel_ = new KernelAB_SOA_Unroll(lattice_, geometry_,q_);
#else
         kernel_ = new KernelAB_SOA_Unroll(lattice_, geometry_);
#endif
      }
   }
   else
   {
      if (parameters_.getPullSetting() == 1)
      {
#ifdef USE_SYCL
	kernel_ = new KernelAB_SOA_Pull(lattice_, geometry_,q_);
#else
         kernel_ = new KernelAB_SOA_Pull(lattice_, geometry_);
#endif
      }
      else
      {
#ifdef USE_SYCL
	kernel_ = new KernelAB_SOA_Base(lattice_, geometry_,q_);
#else
         kernel_ = new KernelAB_SOA_Base(lattice_, geometry_);
#endif
      }
   }
#elif AOS
#ifdef USE_SYCL
	kernel_ = new KernelAB_AOS_Base(lattice_, geometry_,q_);
#else
   kernel_ = new KernelAB_AOS_Base(lattice_, geometry_);
#endif
#endif
   kernel_->setup();

   
   return;
}

/******************************************************************************/
void PropagationAB::initialize(void)
{

   int nFluid = geometry_.getNumFluidPts();
   
#ifdef USE_KOKKOS
   myViewPDF weight("weight",_LATTICESIZE_);
   myMirrorViewPDF weight_h = Kokkos::create_mirror_view(weight);
   for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
   {
      weight_h(stencilIdx) = lattice_.stencil_[stencilIdx].c;
   }
   Kokkos::deep_copy(weight, weight_h);
   Kokkos::fence();
   Kokkos::parallel_for(myPolicy(0, nFluid),KOKKOS_CLASS_LAMBDA (const int fluidIdx)
   {
   	  for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
#ifdef SOA
         dstrb_d_(nFluid * stencilIdx + fluidIdx) = weight(stencilIdx);
         dstrb2_d_(nFluid * stencilIdx + fluidIdx) = weight(stencilIdx);
#elif AOS
         dstrb_d_(_LATTICESIZE_ * fluidIdx + stencilIdx) = weight(stencilIdx);
         dstrb2_d_(_LATTICESIZE_ * fluidIdx + stencilIdx) = weight(stencilIdx);
#endif      	  
      }
   });
   Kokkos::fence();
   
#else
   Pdf* weight;
   weight = new Pdf[_LATTICESIZE_];


   for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
   {
      weight[stencilIdx] = lattice_.stencil_[stencilIdx].c;
   }

   for (int fluidIdx=0; fluidIdx<nFluid; fluidIdx++)

   {
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
#ifdef SOA
         dstrb_[nFluid * stencilIdx + fluidIdx] = weight[stencilIdx];
         dstrb2_[nFluid * stencilIdx + fluidIdx] = weight[stencilIdx];
#elif AOS
         dstrb_[_LATTICESIZE_ * fluidIdx + stencilIdx] = weight[stencilIdx];
         dstrb2_[_LATTICESIZE_ * fluidIdx + stencilIdx] = weight[stencilIdx];
#endif
      }
   }
#ifdef USE_SYCL
	q_.memcpy(dstrb_d_,dstrb_,nFluid *_LATTICESIZE_*sizeof(Pdf));
	q_.memcpy(dstrb2_d_,dstrb2_,nFluid *_LATTICESIZE_*sizeof(Pdf));
	q_.wait();
#endif

#endif
   return;
}

/******************************************************************************/
void PropagationAB::run(void)
{
   initialize();

   int steps = parameters_.getTimesteps();
   double startTime = 0;

   MPI_Barrier(MPI_COMM_WORLD);
   startTime = MPI_Wtime();
#ifdef USE_KOKKOS
      for (int t=0; t<steps; t++)
      {
         kernel_->timestepForce(dstrb_d_, dstrb2_d_, geometry_.getBorderStart(), geometry_.getBorderCount());
         kernel_->timestepForce(dstrb_d_, dstrb2_d_, geometry_.getBulkStart(), geometry_.getBulkCount());
         comm_.exchange(dstrb2_d_.data());
         swap(dstrb_d_, dstrb2_d_);
      }
#elif defined(USE_SYCL)
	for (int t=0; t<steps; t++)
      {
         kernel_->timestepForce(dstrb_d_, dstrb2_d_, geometry_.getBorderStart(), geometry_.getBorderCount());
         kernel_->timestepForce(dstrb_d_, dstrb2_d_, geometry_.getBulkStart(), geometry_.getBulkCount());
         comm_.exchange(dstrb2_d_);
         swap(dstrb_d_, dstrb2_d_);
      }
#else
   
      for (int t=0; t<steps; t++)
      {
         kernel_->timestepForce(dstrb_, dstrb2_, geometry_.getBorderStart(), geometry_.getBorderCount());
         kernel_->timestepForce(dstrb_, dstrb2_, geometry_.getBulkStart(), geometry_.getBulkCount());
         comm_.exchange(dstrb2_);
         swap(dstrb_, dstrb2_);
      }
   
#endif
#ifdef USE_SYCL
      q_.wait();
#elif defined(USE_KOKKOS)
      Kokkos::fence();
#endif
   MPI_Barrier(MPI_COMM_WORLD);
   double endTime = MPI_Wtime();
   
#ifdef USE_KOKKOS
   Kokkos::deep_copy(dstrb_h_,dstrb_d_);
#endif
#if defined(USE_SYCL)
	q_.memcpy(dstrb_,dstrb_d_,geometry_.getNumFluidPts()*_LATTICESIZE_*sizeof(Pdf));
	q_.wait();
#endif
   if (myRank_ == 0)
   {
      printMFLUPS(endTime-startTime);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   checkResult(steps-1);
   if(parameters_.getWriteVtk())
   {
      writeVTK(steps-1);
   }

   return;
}

/******************************************************************************/
Pdf PropagationAB::getRho(int fluidIdx, int currentStep)
{
   Pdf rho = 0.0;
   int nFluid = geometry_.getNumFluidPts();
   int dataLoc = _INVALID_;
   for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
   {
#ifdef SOA
      dataLoc = nFluid * stencilIdx + fluidIdx;
#elif AOS
      dataLoc = _LATTICESIZE_ * fluidIdx + stencilIdx;
#endif

#ifdef USE_KOKKOS
	  rho += dstrb_h_(dataLoc);
#else
      rho += dstrb_[dataLoc];
#endif
   }
   return rho;
}

/******************************************************************************/
vector<Pdf> PropagationAB::getVelocity(int fluidIdx, int currentStep)
{
   Pdf rho = 0.0;
   vector<Pdf> vel(_DIMS_, 0.0);
   int nFluid = geometry_.getNumFluidPts();
   int dataLoc = _INVALID_;

   for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
   {
#ifdef SOA
      dataLoc = nFluid * stencilIdx + fluidIdx;
#elif AOS
      dataLoc = _LATTICESIZE_ * fluidIdx + stencilIdx;
#endif
#ifdef USE_KOKKOS
	  rho += dstrb_h_(dataLoc);
#else
      rho += dstrb_[dataLoc];
#endif
      for (int dim=0; dim<_DIMS_; dim++)
      {
#ifdef USE_KOKKOS
	  	 vel[dim] += dstrb_h_(dataLoc) * lattice_.stencil_[stencilIdx].dir[dim];
#else
         vel[dim] += dstrb_[dataLoc] * lattice_.stencil_[stencilIdx].dir[dim];
#endif
      }
   }

   Pdf invRho = 1.0/rho;
   for (int dim=0; dim<_DIMS_; dim++)
   {
      vel[dim] *= invRho;
   }

   return vel;
}

/******************************************************************************/
