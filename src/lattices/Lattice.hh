#ifndef LATTICE_H
#define LATTICE_H

#include <vector>

#include "Parameters.hh"

using namespace::std;

typedef int LatticeT;

typedef struct {
   int opp;
   int dir[3];
   double c;
} Stencil;

class Lattice
{
   public:

      Lattice(void) {};
      virtual ~Lattice(void) {};
      int getVel(int velIdx, int dim) { return stencil_[velIdx].dir[dim]; }
      int getOpp(int velIdx) { return stencil_[velIdx].opp; }

      vector<Stencil> stencil_;
      double inv_cs2, cs2;
      int interactDistance;

   protected:



};
#endif
