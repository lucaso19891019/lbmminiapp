#include <iostream>
#include <mpi.h>

#include "KernelAA_AOS_Base.hh"
#include "KernelAA_SOA_Base.hh"
#include "KernelAA_SOA_Unroll.hh"
#include "PropagationAA.hh"

/******************************************************************************/
void PropagationAA::setup(void)
{
   // initialize storage arrays
   commOffset_ = comm_.getCommLength();
   int dstrbSize = geometry_.getNumFluidPts() * _LATTICESIZE_;
#if defined(USE_KOKKOS)
   dstrb_d_= myViewPDF("dstrb_d_",dstrbSize);
   dstrb_h_ = Kokkos::create_mirror_view(dstrb_d_);
   
#else
#if defined (USE_SYCL)
	dstrb_d_=sycl::malloc_device<Pdf>(dstrbSize,q_);
#endif
   dstrb_ = new Pdf[dstrbSize];
   fill(dstrb_, dstrb_+dstrbSize, 0.0);
#endif

   // select kernel
   kernel_ = 0;
#ifdef SOA
   if (parameters_.getUnrollSetting() == 1 && _LATTICESIZE_ == 19)
   {
#if defined (USE_SYCL)
	kernel_ = new KernelAA_SOA_Unroll(lattice_, geometry_,q_);
#else
      kernel_ = new KernelAA_SOA_Unroll(lattice_, geometry_);
#endif
   }
   else
   {
#if defined (USE_SYCL)
	kernel_ = new KernelAA_SOA_Base(lattice_, geometry_,q_);
#else
      kernel_ = new KernelAA_SOA_Base(lattice_, geometry_);
#endif
   }
#elif AOS
#if defined (USE_SYCL)
	kernel_ = new KernelAA_AOS_Base(lattice_, geometry_,q_);
#else
   kernel_ = new KernelAA_AOS_Base(lattice_, geometry_);
#endif
#endif
   kernel_->setup();

   
   return;
}

/******************************************************************************/
void PropagationAA::initialize(void)
{

   int nFluid = geometry_.getNumFluidPts();

#ifdef USE_KOKKOS
   myViewPDF weight("weight",_LATTICESIZE_);
   myViewInt opp("opp",_LATTICESIZE_);
   myMirrorViewPDF weight_h = Kokkos::create_mirror_view(weight);
   myMirrorViewInt opp_h = Kokkos::create_mirror_view(opp);
   for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
   {
      weight_h(stencilIdx) = lattice_.stencil_[stencilIdx].c;
      opp_h(stencilIdx) = lattice_.getOpp(stencilIdx);
   }
   Kokkos::deep_copy(weight, weight_h);
   Kokkos::deep_copy(opp, opp_h);
   Kokkos::fence();
   Kokkos::parallel_for(myPolicy(0, nFluid),KOKKOS_CLASS_LAMBDA (const int fluidIdx)
   {
   	  for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
#ifdef SOA
         dstrb_d_(nFluid * stencilIdx + fluidIdx) = weight(stencilIdx);
#elif AOS
         dstrb_d_(_LATTICESIZE_ * fluidIdx + stencilIdx) = weight(stencilIdx);
 #endif      	  
      }
   });
   Kokkos::fence();
   
#else

   Pdf* weight;
   int* opp;
   weight = new Pdf[_LATTICESIZE_];
   opp = new int[_LATTICESIZE_];

   for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
   {
      weight[stencilIdx] = lattice_.stencil_[stencilIdx].c;
      opp[stencilIdx] = lattice_.getOpp(stencilIdx);
   }

   for (int fluidIdx=0; fluidIdx<nFluid; fluidIdx++)

   {
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
         int oppIdx = opp[stencilIdx];
#ifdef SOA
         dstrb_[nFluid * oppIdx + fluidIdx] = weight[stencilIdx];
#elif AOS
         dstrb_[_LATTICESIZE_ * fluidIdx + oppIdx] = weight[stencilIdx];
#endif
      }
   }
#ifdef USE_SYCL
   q_.memcpy(dstrb_d_,dstrb_,nFluid*_LATTICESIZE_*sizeof(Pdf));
	q_.wait();
#endif
#endif
   return;
}

/******************************************************************************/
void PropagationAA::run(void)
{

   initialize();

   int steps = parameters_.getTimesteps();

   MPI_Barrier(MPI_COMM_WORLD);
   double startTime = MPI_Wtime();

#ifdef USE_KOKKOS
      for (int t=0; t<steps; t++)
      {
         if (t%2 == 0)
         {
            kernel_->timestepEvenForce(dstrb_d_, geometry_.getInletStart(), geometry_.getInletCount()+geometry_.getOutletCount()+geometry_.getBorderCount());
            kernel_->timestepEvenForce(dstrb_d_, geometry_.getBulkStart(), geometry_.getBulkCount());
            comm_.exchange(dstrb_d_.data());
         }
         else
         {
            kernel_->timestepOddForce(dstrb_d_, geometry_.getInletStart(), geometry_.getInletCount()+geometry_.getOutletCount()+geometry_.getBorderCount());
            kernel_->timestepOddForce(dstrb_d_, geometry_.getBulkStart(), geometry_.getBulkCount());
            comm_.exchange(dstrb_d_.data());
         }
      }
#elif defined(USE_SYCL)
	  for (int t=0; t<steps; t++)
      {
         if (t%2 == 0)
         {
            kernel_->timestepEvenForce(dstrb_d_, geometry_.getInletStart(), geometry_.getInletCount()+geometry_.getOutletCount()+geometry_.getBorderCount());
            kernel_->timestepEvenForce(dstrb_d_, geometry_.getBulkStart(), geometry_.getBulkCount());
            comm_.exchange(dstrb_d_);
         }
         else
         {
            kernel_->timestepOddForce(dstrb_d_, geometry_.getInletStart(), geometry_.getInletCount()+geometry_.getOutletCount()+geometry_.getBorderCount());
            kernel_->timestepOddForce(dstrb_d_, geometry_.getBulkStart(), geometry_.getBulkCount());
            comm_.exchange(dstrb_d_);
         }
      }
#else
      for (int t=0; t<steps; t++)
      {
         if (t%2 == 0)
         {
            kernel_->timestepEvenForce(dstrb_, geometry_.getInletStart(), geometry_.getInletCount()+geometry_.getOutletCount()+geometry_.getBorderCount());
            kernel_->timestepEvenForce(dstrb_, geometry_.getBulkStart(), geometry_.getBulkCount());
            comm_.exchange(dstrb_);
         }
         else
         {
            kernel_->timestepOddForce(dstrb_, geometry_.getInletStart(), geometry_.getInletCount()+geometry_.getOutletCount()+geometry_.getBorderCount());
            kernel_->timestepOddForce(dstrb_, geometry_.getBulkStart(), geometry_.getBulkCount());
            comm_.exchange(dstrb_);
         }
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
   checkResult(steps);
   if(parameters_.getWriteVtk())
   {
      writeVTK(steps);
   }


   return;
}

/******************************************************************************/
Pdf PropagationAA::getRho(int fluidIdx, int currentStep)
{
   Pdf rho = 0.0;
   int nFluid = geometry_.getNumFluidPts();
   int dataLoc = _INVALID_;

   if (currentStep%2 == 0)
   {
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
#ifdef SOA
         dataLoc = nFluid * lattice_.getOpp(stencilIdx) + fluidIdx;
#elif AOS
         dataLoc = _LATTICESIZE_ * fluidIdx + lattice_.getOpp(stencilIdx);
#endif
#ifdef USE_KOKKOS
           rho += dstrb_h_(dataLoc);
#else
      rho += dstrb_[dataLoc];
#endif

      }
   }
   else
   {
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
#ifdef SOA
         dataLoc = geometry_.adjacency_[nFluid * stencilIdx + fluidIdx];
#elif AOS
         dataLoc = geometry_.adjacency_[_LATTICESIZE_ * fluidIdx + stencilIdx];
#endif
#ifdef USE_KOKKOS
           rho += dstrb_h_(dataLoc);
#else
      rho += dstrb_[dataLoc];
#endif

      }
   }
   return rho;
}

/******************************************************************************/
vector<Pdf> PropagationAA::getVelocity(int fluidIdx, int currentStep)
{
   Pdf rho = 0.0;
   vector<Pdf> vel(_DIMS_, 0.0);
   int nFluid = geometry_.getNumFluidPts();
   int dataLoc = _INVALID_;

   if (currentStep%2 == 0)
   {
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
#ifdef SOA
         dataLoc = nFluid * lattice_.getOpp(stencilIdx) + fluidIdx;
#elif AOS
         dataLoc = _LATTICESIZE_ * fluidIdx + lattice_.getOpp(stencilIdx);
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
   }
   else
   {
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
#ifdef SOA
         dataLoc = geometry_.adjacency_[nFluid * stencilIdx + fluidIdx];
#elif AOS
         dataLoc = geometry_.adjacency_[_LATTICESIZE_ * fluidIdx + stencilIdx];
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
   }

   Pdf invRho = 1.0/rho;
   for (int dim=0; dim<_DIMS_; dim++)
   {
      vel[dim] *= invRho;
   }

   return vel;
}

/******************************************************************************/
