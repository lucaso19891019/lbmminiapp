
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>
#include <mpi.h>
#include <sstream>

#include "Propagation.hh"

/******************************************************************************/
#ifdef USE_SYCL
Propagation::Propagation(Lattice& lattice, Geometry& geometry, Communication& comm,
   Parameters& parameters,sycl::queue& q) : lattice_(lattice), geometry_(geometry), comm_(comm),
   parameters_(parameters),q_(q)
#else
Propagation::Propagation(Lattice& lattice, Geometry& geometry, Communication& comm,
   Parameters& parameters) : lattice_(lattice), geometry_(geometry), comm_(comm),
   parameters_(parameters)
#endif
{

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank_);
   MPI_Comm_size(MPI_COMM_WORLD, &nTasks_);

}

/******************************************************************************/
Propagation::~Propagation(void)
{

}

/******************************************************************************/
void Propagation::writeVTK(int currentStep)
{

   ostringstream oss;
   oss.width(5);
   oss.fill('0');
   oss << myRank_;

   ofstream toFile;
   string nameFile = "outs." + oss.str() + ".vtk";
   toFile.open(nameFile.c_str());
   toFile << "# vtk DataFile Version 3.0" << endl;
   toFile << "Placeholder" << endl;
   toFile << "ASCII" << endl;
   toFile << "DATASET STRUCTURED_POINTS" << endl;
   toFile << "DIMENSIONS " << geometry_.getDimsByDim(0) << " "
                           << geometry_.getDimsByDim(1) << " "
                           << geometry_.getDimsByDim(2) << endl;
   toFile << "ORIGIN " << geometry_.getMinByDim(0) << " "
                       << geometry_.getMinByDim(1) << " "
                       << geometry_.getMinByDim(2) << endl;
   toFile << "SPACING 1 1 1" << endl;
   toFile << "POINT_DATA " << geometry_.getDimsByDim(0) *
                              geometry_.getDimsByDim(1) *
                              geometry_.getDimsByDim(2) << endl;

   // print rho
   {
      toFile << "SCALARS rho double" << endl;
      toFile << "LOOKUP_TABLE default" << endl;
      for (int idxK=0; idxK<geometry_.getDimsByDim(2); idxK++)
      {
         int myK = geometry_.getMinByDim(2) + idxK;
         for (int idxJ=0; idxJ<geometry_.getDimsByDim(1); idxJ++)
         {
            int myJ = geometry_.getMinByDim(1) + idxJ;
            for (int idxI=0; idxI<geometry_.getDimsByDim(0); idxI++)
            {
               int myI = geometry_.getMinByDim(0) + idxI;
               int myIJK = geometry_.getGridIdx(myI, myJ, myK);
               Pdf rho = 0.0;
               if(myIJK >= 0)
               {
                  int fluidIdx = geometry_.getGridElement(myIJK);
                  if (fluidIdx >= 0)
                  {
                     rho = getRho(fluidIdx, currentStep);
                  }
               }
               toFile << rho << endl;
            }
         }
      }
   }

   // print lattice point types
   {
      toFile << "SCALARS id int" << endl;
      toFile << "LOOKUP_TABLE default" << endl;
      for (int idxK=0; idxK<geometry_.getDimsByDim(2); idxK++)
      {
         int myK = geometry_.getMinByDim(2) + idxK;
         for (int idxJ=0; idxJ<geometry_.getDimsByDim(1); idxJ++)
         {
            int myJ = geometry_.getMinByDim(1) + idxJ;
            for (int idxI=0; idxI<geometry_.getDimsByDim(0); idxI++)
            {
               int myI = geometry_.getMinByDim(0) + idxI;
               int myIJK = geometry_.getGridIdx(myI, myJ, myK);
               int fluidIdx = geometry_.getGridElement(myIJK);
               toFile << fluidIdx << endl;
            }
         }
      }
   }

   // print velocity
   {
      toFile << "VECTORS velocity double" << endl;
      for (int idxK=0; idxK<geometry_.getDimsByDim(2); idxK++)
      {
         int myK = geometry_.getMinByDim(2) + idxK;
         for (int idxJ=0; idxJ<geometry_.getDimsByDim(1); idxJ++)
         {
            int myJ = geometry_.getMinByDim(1) + idxJ;
            for (int idxI=0; idxI<geometry_.getDimsByDim(0); idxI++)
            {
               int myI = geometry_.getMinByDim(0) + idxI;
               int myIJK = geometry_.getGridIdx(myI, myJ, myK);
               vector<Pdf> vel(_DIMS_, 0.0);
               if(myIJK >= 0)
               {
                  int fluidIdx = geometry_.getGridElement(myIJK);
                  if (fluidIdx >= 0)
                  {
                     vel = getVelocity(fluidIdx, currentStep);
                  }
               }
               toFile << vel[0] << " " << vel[1] << " " << vel[2] << endl;
            }
         }
      }
   }

   // print coordinates
   {
      toFile << "VECTORS nodes int" << endl;
      for (int idxK=0; idxK<geometry_.getDimsByDim(2); idxK++)
      {
         int myK = geometry_.getMinByDim(2) + idxK;
         for (int idxJ=0; idxJ<geometry_.getDimsByDim(1); idxJ++)
         {
            int myJ = geometry_.getMinByDim(1) + idxJ;
            for (int idxI=0; idxI<geometry_.getDimsByDim(0); idxI++)
            {
               int myI = geometry_.getMinByDim(0) + idxI;
               toFile << myI << " " << myJ << " " << myK << endl;
            }
         }
      }
   }

   toFile.close();

   return;
}

/******************************************************************************/
void Propagation::checkResult(int currentStep)
{

   int centerI = (geometry_.getTotalLengthByDim(0)-1)/2;
   int centerJ = (geometry_.getTotalLengthByDim(1)-1)/2;
   int centerK = (geometry_.getTotalLengthByDim(2)-1)/2;
   int centerIJK = geometry_.getGridIdx(centerI, centerJ, centerK);
   if (centerIJK > _INVALID_)
   {
      int fluidIdx = geometry_.getGridElement(centerIJK);
      if (fluidIdx > _INVALID_)
      {
         
            Pdf diameter = 16.0*parameters_.getSimulationSize();
            Pdf nu = _ONETHIRD_ * (1.0/_OMEGA_ - 0.5);
            Pdf rhoComputed = getRho(fluidIdx, currentStep);
            vector<Pdf> vel = getVelocity(fluidIdx, currentStep);
            Pdf velComputed = vel[0];
            Pdf velExpected = (_GRAVITY_*diameter*diameter) / (16.0 * rhoComputed * nu);
            Pdf tolerance = 0.25;
            if ( (velComputed-velExpected)/velExpected < tolerance)
            {
               cout << "Passed baseline validation" << endl;
            }
            else
            {
               cout << "Failed baseline validation" << endl;
            }
         
      }
   }

   return;
}

/******************************************************************************/
void Propagation::printMFLUPS(double runtime)
{
   // MFEUPS definition from Ansumali (https://doi.org/10.1103/PhysRevE.88.013314)

   double mflups = (geometry_.getGlobalNumFluidPts() * parameters_.getTimesteps()) / (1000000 * runtime);
   double mfeups = mflups * (_LATTICESIZE_-1);
   cout << "Overall performance" << endl;
   cout << "Total " << mfeups << " MFEUPS (million fluid elementary updates per second)" << endl;
   cout << "Total " << mflups << " MFLUPS (million fluid lattice updates per second)" << endl;
   cout << "Avg " << mflups/nTasks_ << " MFLUPS per rank" << endl;

   return;
}

/******************************************************************************/