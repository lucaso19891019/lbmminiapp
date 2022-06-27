#ifndef KERNEL_AA_SOA_BASE_H
#define KERNEL_AA_SOA_BASE_H


#include "KernelAA.hh"

class KernelAA_SOA_Base : public KernelAA
{
   public:
#ifdef USE_SYCL
	KernelAA_SOA_Base(Lattice& lattice, Geometry& geometry,sycl::queue& q): KernelAA(lattice, geometry,q) {};
#else
      KernelAA_SOA_Base(Lattice& lattice, Geometry& geometry): KernelAA(lattice, geometry) {};
#endif
      ~KernelAA_SOA_Base(void) {};

      void setup();
#ifdef USE_KOKKOS
      virtual void timestepEvenForce(myViewPDF dstrb, int startIdx, int countIdx) ;
      virtual void timestepOddForce(myViewPDF dstrb, int startIdx, int countIdx) ;
#else
      virtual void timestepEvenForce(Pdf* dstrb, int startIdx, int countIdx) ;
      virtual void timestepOddForce(Pdf* dstrb, int startIdx, int countIdx) ;

#endif

   protected:



   private:



};
#endif
