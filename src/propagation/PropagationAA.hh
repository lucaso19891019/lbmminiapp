#ifndef PROPAGATION_AA_H
#define PROPAGATION_AA_H

#include "KernelAA.hh"
#include "Propagation.hh"


using namespace::std;

class PropagationAA : public Propagation
{
   public:
#ifdef USE_SYCL
	PropagationAA(Lattice& lattice, Geometry& geometry, Communication& comm,
         Parameters& parameters,sycl::queue& q) : Propagation(lattice, geometry, comm, parameters,q) {};
#else
      PropagationAA(Lattice& lattice, Geometry& geometry, Communication& comm,
         Parameters& parameters) : Propagation(lattice, geometry, comm, parameters) {};
#endif
      ~PropagationAA(void) {
#ifdef USE_SYCL
	sycl::free(dstrb_d_,q_);
#endif
};
      void setup(void);
      void initialize(void);
      void run(void);
      Pdf getRho(int fluidIdx, int currentStep);
      vector<Pdf> getVelocity(int fluidIdx, int currentStep);

   protected:

      KernelAA* kernel_;

};
#endif
