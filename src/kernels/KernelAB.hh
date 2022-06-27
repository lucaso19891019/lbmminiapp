#ifndef KERNEL_AB_H
#define KERNEL_AB_H

#include "Geometry.hh"
#include "Lattice.hh"
#include "Parameters.hh"

class KernelAB
{
   public:
#if defined(USE_SYCL)
      KernelAB(Lattice& lattice, Geometry& geometry,sycl::queue& q): lattice_(lattice), geometry_(geometry),q_(q) {};
#else
      KernelAB(Lattice& lattice, Geometry& geometry): lattice_(lattice), geometry_(geometry) {};
#endif
      virtual ~KernelAB(void) {
#if defined(USE_SYCL)
	sycl::free(stencil_d_,q_);
	sycl::free(weight_d_,q_);
#endif
};

      virtual void setup() = 0;
#ifdef USE_KOKKOS
         virtual void timestepForce(myViewPDF dstrb_src, myViewPDF dstrb_tgt, int startIdx, int countIdx) = 0;
#else
    
      virtual void timestepForce(Pdf* dstrb_src, Pdf* dstrb_tgt, int startIdx, int countIdx) = 0;
#endif
   protected:

      Lattice& lattice_;
      Geometry& geometry_;
#ifdef USE_KOKKOS
      myViewPDF stencil_d_, weight_d_;
      myMirrorViewPDF stencil_h_, weight_h_;
#endif
#ifdef USE_SYCL
	sycl::queue& q_;
      Pdf* stencil_d_;
      Pdf* weight_d_;
#endif
      Pdf* stencil_;
      Pdf* weight_;

   private:



};
#endif
