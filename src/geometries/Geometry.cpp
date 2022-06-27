
#include <cassert>
#include <iostream>
#include <mpi.h>
#include <math.h>

#include "Geometry.hh"

/******************************************************************************/
#if defined (USE_SYCL)
Geometry::Geometry(Lattice& lattice, Parameters& parameters,sycl::queue& q) :
   lattice_(lattice), parameters_(parameters),q_(q)
#else
Geometry::Geometry(Lattice& lattice, Parameters& parameters) :
   lattice_(lattice), parameters_(parameters)
#endif
{
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank_);
   MPI_Comm_size(MPI_COMM_WORLD, &nTasks_);
}

/******************************************************************************/
void Geometry::decomposeDomain(void)
{
   // nb: all lengths are denoted in lattice units

   // temporary placeholder
   boxes_.resize(_DIMS_, 1);
   if (nTasks_ % 3 == 0)
   {
      int pow2Tasks = nTasks_/3;
      int logTasks = log2(pow2Tasks);
      assert(pow(2, logTasks) == pow2Tasks);
      boxes_[0] *= 3;
      for (int boxIdx = 0; boxIdx<logTasks; boxIdx++)
      {
         if (boxIdx < 1) { boxes_[0] *= 2; }
         else { boxes_[(boxIdx-1) % 3] *= 2; }
     }
   }
   else
   {
      int logTasks = log2(nTasks_);
      assert(pow(2, logTasks) == nTasks_);
      for (int boxIdx = 0; boxIdx<logTasks; boxIdx++)
      {
         if (boxIdx < 2) { boxes_[0] *= 2; }
         else { boxes_[(boxIdx-2) % 3] *= 2; }
      }
   }

   // assign tasks with z fastest, y next, and x slowest
   vector<int> myBox(_DIMS_, _INVALID_);
   myBox[0] = myRank_ / (boxes_[1] * boxes_[2]);
   myBox[1] = (myRank_ - myBox[0] * boxes_[1] * boxes_[2]) / boxes_[2];
   myBox[2] = myRank_ - myBox[0] * boxes_[1] * boxes_[2] - myBox[1] * boxes_[2];

   // Geometry Cases
   int geometryLibID = parameters_.getGeometryLibID();
   if (geometryLibID == 1)    // geometryID 1: cylinder
   {
      // compute in double space and cast to ints
      double simulationSize = parameters_.getSimulationSize();
      int radius = 8.0 * simulationSize;
      int axialLength = 84.0 * simulationSize;

      totalLength_.resize(_DIMS_, _INVALID_);
      totalLength_[0] = axialLength;
      totalLength_[1] = 2*radius+3;
      totalLength_[2] = 2*radius+3;
   }

   vector<int> taskLength = {_INVALID_, _INVALID_, _INVALID_};
   for (int dim=0; dim<_DIMS_; dim++)
   {
      // ceiling arithmetic
      taskLength[dim] = ( totalLength_[dim] + boxes_[dim] - 1 ) / boxes_[dim];
   }

   // Min and max values owned by task *inclusive of both values*
   myMin_.resize(_DIMS_, _INVALID_);
   myMax_.resize(_DIMS_, _INVALID_);
   for (int dim=0; dim<_DIMS_; dim++)
   {
      myMin_[dim] = myBox[dim] * taskLength[dim];
      if (myBox[dim] != boxes_[dim]-1)
      {
         myMax_[dim] = (myBox[dim]+1) * taskLength[dim] - 1;
      }
      else
      {
         myMax_[dim] = totalLength_[dim] - 1;
      }
   }

   return;
}

/******************************************************************************/
void Geometry::setupGrid(void)
{
   myDims_ = {_INVALID_, _INVALID_, _INVALID_};
   for (int dim=0; dim<_DIMS_; dim++)
   {
      myDims_[dim] = myMax_[dim] - myMin_[dim] + 1;
   }

   grid_.resize(myDims_[0] * myDims_[1] * myDims_[2]);

   // Geometry Cases

   int geometryLibID = parameters_.getGeometryLibID();
   if (geometryLibID == 1)    // geometryID 1: cylinder
   {
      double radius = (totalLength_[1]-3)/2;
      double axisCenter = radius + 1.5;
      for (int idxI=0; idxI<myDims_[0]; idxI++)
      {
         int myI = myMin_[0] + idxI;
         for (int idxJ=0; idxJ<myDims_[1]; idxJ++)
         {
            int myJ = myMin_[1] + idxJ;
            for (int idxK=0; idxK<myDims_[2]; idxK++)
            {
               int myK = myMin_[2] + idxK;
               int idxIJK = myDims_[1] * myDims_[2] * idxI + myDims_[2] * idxJ + idxK;
               int dist = sqrt( pow(myJ-axisCenter, 2.0) + pow(myK-axisCenter, 2.0));
               if (dist <= radius )
               {
                  vector<int> loc = {myI, myJ, myK};
                  fluidPts_.push_back(loc);
               }
               else
               {
                  grid_[idxIJK] = LAT_PT_WALL;
               }
            }
         }
      }
   }

   nFluid_ = fluidPts_.size();
   nFluidGlobal_ = nFluid_;
   MPI_Allreduce(MPI_IN_PLACE, &nFluidGlobal_, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
   if (myRank_ == 0)
   {
      cout << "Simulation has " << nFluidGlobal_ << " fluid points" << endl;
   }

   return;
}

/******************************************************************************/
void Geometry::assignFluidIndices()
{

   for (int fluidIdx=0; fluidIdx<nFluid_; fluidIdx++)
   {
      int myI = fluidPts_[fluidIdx][0];
      int myJ = fluidPts_[fluidIdx][1];
      int myK = fluidPts_[fluidIdx][2];
      int idxIJK = getGridIdx(myI, myJ, myK);
      grid_[idxIJK] = fluidIdx;
   }

   return;
}

/******************************************************************************/
int Geometry::getGridIdx(int i, int j, int k)
{
   int relI = i - myMin_[0];
   int relJ = j - myMin_[1];
   int relK = k - myMin_[2];
   // jpg: this safeguard is currently unnecessary.
   if (relI < 0 || relJ < 0 || relK < 0)
   {
      return _INVALID_;
   }
   if (relI >= myDims_[0] || relJ >= myDims_[1] || relK >= myDims_[2])
   {
      return _INVALID_;
   }
   return myDims_[2] * myDims_[1] * relI + myDims_[2] * relJ + relK;
}

/******************************************************************************/
