#ifndef KERNEL_AA_H
#define KERNEL_AA_H

#include "Geometry.hh"
#include "Lattice.hh"

class KernelAA
{
   public:
#if defined(USE_SYCL)
      KernelAA(Lattice& lattice, Geometry& geometry,sycl::queue& q): lattice_(lattice), geometry_(geometry),q_(q) {};
#else
	KernelAA(Lattice& lattice, Geometry& geometry): lattice_(lattice), geometry_(geometry) {};
#endif
      virtual ~KernelAA(void) {
#if defined(USE_SYCL)
	sycl::free(stencil_d_,q_);
	sycl::free(weight_d_,q_);
	sycl::free(opp_d_,q_);
#endif
};

      virtual void setup() = 0;
#ifdef USE_KOKKOS
      virtual void timestepEvenForce(myViewPDF dstrb, int startIdx, int countIdx) = 0;
      virtual void timestepOddForce(myViewPDF dstrb, int startIdx, int countIdx) = 0;
#else
      virtual void timestepEvenForce(Pdf* dstrb, int startIdx, int countIdx) = 0;
      virtual void timestepOddForce(Pdf* dstrb, int startIdx, int countIdx) = 0;

#endif

   protected:

      Lattice& lattice_;
      Geometry& geometry_;
#ifdef USE_KOKKOS
      myViewPDF stencil_d_, weight_d_;
      myViewInt opp_d_;
      myMirrorViewPDF stencil_h_, weight_h_;
      myMirrorViewInt opp_h_;
#endif
#ifdef USE_SYCL
	sycl::queue& q_;
      Pdf* stencil_d_;
      Pdf* weight_d_;
      int* opp_d_;
#endif

      Pdf* stencil_;
      Pdf* weight_;
      int* opp_;

   private:



};
#endif
