#ifndef PROPAGATION_AB_H
#define PROPAGATION_AB_H

#include "KernelAB.hh"
#include "Propagation.hh"

using namespace::std;

class PropagationAB : public Propagation
{
   public:
#ifdef USE_SYCL
	PropagationAB(Lattice& lattice, Geometry& geometry, Communication& comm,
         Parameters& parameters,sycl::queue& q) : Propagation(lattice, geometry, comm, parameters,q) {};
#else
      PropagationAB(Lattice& lattice, Geometry& geometry, Communication& comm,
         Parameters& parameters) : Propagation(lattice, geometry, comm, parameters) {};
#endif
      ~PropagationAB(void) {
#ifdef USE_SYCL
	sycl::free(dstrb_d_,q_);
	sycl::free(dstrb2_d_,q_);
#endif
};
      void setup(void);
      void initialize(void);
      void run(void);
      Pdf getRho(int fluidIdx, int currentStep);
      vector<Pdf> getVelocity(int fluidIdx, int currentStep);

   protected:

      KernelAB* kernel_;

};
#endif
