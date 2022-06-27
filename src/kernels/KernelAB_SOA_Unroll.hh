#ifndef KERNEL_AB_SOA_UNROLL_H
#define KERNEL_AB_SOA_UNROLL_H



#include "KernelAB.hh"

class KernelAB_SOA_Unroll : public KernelAB
{
   public:
#ifdef USE_SYCL
	KernelAB_SOA_Unroll(Lattice& lattice, Geometry& geometry,sycl::queue& q): KernelAB(lattice, geometry,q) {};
#else
      KernelAB_SOA_Unroll(Lattice& lattice, Geometry& geometry): KernelAB(lattice, geometry) {};
#endif
      ~KernelAB_SOA_Unroll(void) {};

      void setup();
#ifdef USE_KOKKOS
      
      void timestepForce(myViewPDF dstrb_src, myViewPDF dstrb_tgt, int startIdx, int countIdx);
#else
           void timestepForce(Pdf* dstrb_src, Pdf* dstrb_tgt, int startIdx, int countIdx);
#endif

   protected:



   private:



};
#endif
