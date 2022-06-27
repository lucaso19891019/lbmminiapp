#ifndef PROPAGATION_H
#define PROPAGATION_H

#include "Communication.hh"
#include "Geometry.hh"
#include "Lattice.hh"
#include "Parameters.hh"

class Propagation
{
   public:
#ifdef USE_SYCL
	Propagation(Lattice& lattice, Geometry& geometry, Communication& comm, Parameters& parameters, sycl::queue& q);
#else
      Propagation(Lattice& lattice, Geometry& geometry, Communication& comm, Parameters& parameters);
#endif
      virtual ~Propagation(void);
      virtual void setup(void) = 0;
      virtual void initialize(void) = 0;
      virtual void run(void) = 0;
      virtual Pdf getRho(int fluidIdx, int currentStep) = 0;
      virtual vector<Pdf> getVelocity(int fluidIdx, int currentStep) = 0;

      void writeVTK(int currentStep);
      void checkResult(int currentStep);
      void printMFLUPS(double runtime);

   protected:

      Lattice& lattice_;
      Geometry& geometry_;
      Communication& comm_;
      Parameters& parameters_;
#ifdef USE_SYCL
 	sycl::queue& q_;
#endif

      int myRank_, nTasks_;

#ifdef USE_KOKKOS
      myViewPDF dstrb_d_, dstrb2_d_;
      myMirrorViewPDF dstrb_h_;
#endif
#ifdef USE_SYCL
	Pdf *dstrb_d_, *dstrb2_d_;
#endif
      Pdf *dstrb_, *dstrb2_;

      int commOffset_;

   private:



};
#endif
