#ifndef GEOMETRY_AAAB_H
#define GEOMETRY_AAAB_H

#include "Geometry.hh"

using namespace::std;

class GeometryAAAB : public Geometry
{
   public:
#ifdef USE_SYCL
      GeometryAAAB(Lattice& lattice, Parameters& parameters, sycl::queue& q): Geometry(lattice, parameters,q){};
#else
      GeometryAAAB(Lattice& lattice, Parameters& parameters) : Geometry(lattice, parameters) {};
#endif
      ~GeometryAAAB(void) {};

      void setup();
      void setupAdjacency();

      void sortFluidPts();

   private:

      bool checkPtOnTaskBoundary(const vector<int>& fluidPt);
      bool checkPtOnInlet(const vector<int>& fluidPt);
      bool checkPtOnOutlet(const vector<int>& fluidPt);
      int getSortIdxFromPt(const vector<int>& fluidPt);

};
#endif
