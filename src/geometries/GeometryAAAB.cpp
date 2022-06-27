
#include "GeometryAAAB.hh"

#include <iostream>
#include <cassert>
#include <algorithm>

/******************************************************************************/
void GeometryAAAB::setup()
{

   sortFluidPts();

   return;
}

/******************************************************************************/
void GeometryAAAB::setupAdjacency()
{
   int adjacencySize = nFluid_ * _LATTICESIZE_;
   adjacency_ = new int[adjacencySize];
   fill(adjacency_, adjacency_+adjacencySize, _INVALID_);
#ifdef USE_KOKKOS
   adjacency_d_ = myViewInt("adjacency_d",adjacencySize);
   adjacency_h_ = Kokkos::create_mirror_view(adjacency_d_);
#endif
#ifdef USE_SYCL
	adjacency_d_=sycl::malloc_device<int>(adjacencySize,q_);
#endif

   int propPattern = parameters_.getPropPattern();
   int pullSetting = parameters_.getPullSetting();

   for (int fluidIdx=0; fluidIdx<nFluid_; fluidIdx++)
   {
      int myI = fluidPts_[fluidIdx][0];
      int myJ = fluidPts_[fluidIdx][1];
      int myK = fluidPts_[fluidIdx][2];
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
         int nbrI = _INVALID_;
         int nbrJ = _INVALID_;
         int nbrK = _INVALID_;
         if (propPattern == 0 && pullSetting == 0)
         {
            nbrI = myI + lattice_.getVel(stencilIdx, 0);
            nbrJ = myJ + lattice_.getVel(stencilIdx, 1);
            nbrK = myK + lattice_.getVel(stencilIdx, 2);
         }
         else if (propPattern == 1 || pullSetting == 1)
         {
            nbrI = myI + lattice_.getVel(lattice_.getOpp(stencilIdx), 0);
            nbrJ = myJ + lattice_.getVel(lattice_.getOpp(stencilIdx), 1);
            nbrK = myK + lattice_.getVel(lattice_.getOpp(stencilIdx), 2);
         }
         if (nTasks_ == 1)
         {
            if (nbrI < myMin_[0])
            {
               nbrI += myDims_[0];
            }
            else if (nbrI > myMax_[0])
            {
               nbrI -= myDims_[0];
            }
         }
         int nbrIJK = getGridIdx(nbrI, nbrJ, nbrK);
#ifdef SOA
         if(nbrIJK < 0)
         {
            adjacency_[nFluid_ * stencilIdx + fluidIdx] = nFluid_ * lattice_.getOpp(stencilIdx) + fluidIdx;
         }
         else
         {
            int nbrFluidIdx = grid_[nbrIJK];
            if (nbrFluidIdx < 0)
            {
               adjacency_[nFluid_ * stencilIdx + fluidIdx] = nFluid_ * lattice_.getOpp(stencilIdx) + fluidIdx;
            }
            else
            {
               adjacency_[nFluid_ * stencilIdx + fluidIdx] = nFluid_ * stencilIdx + nbrFluidIdx;
            }

         }
         assert(adjacency_[nFluid_ * stencilIdx + fluidIdx] >= 0);
#elif AOS
         if(nbrIJK < 0)
         {
            adjacency_[_LATTICESIZE_ * fluidIdx + stencilIdx] = _LATTICESIZE_ * fluidIdx + lattice_.getOpp(stencilIdx);
         }
         else
         {
            int nbrFluidIdx = grid_[nbrIJK];
            if (nbrFluidIdx < 0)
            {
               adjacency_[_LATTICESIZE_ * fluidIdx + stencilIdx] = _LATTICESIZE_ * fluidIdx + lattice_.getOpp(stencilIdx);
            }
            else
            {
               adjacency_[_LATTICESIZE_ * fluidIdx + stencilIdx] = _LATTICESIZE_ * nbrFluidIdx + stencilIdx;
            }
         }
#endif
#ifdef USE_KOKKOS
#ifdef SOA
		 adjacency_h_(nFluid_ * stencilIdx + fluidIdx) = adjacency_[nFluid_ * stencilIdx + fluidIdx];
#elif AOS
		 adjacency_h_(_LATTICESIZE_ * fluidIdx + stencilIdx) = adjacency_[_LATTICESIZE_ * fluidIdx + stencilIdx];
#endif
#endif
      }
   }

#ifdef USE_KOKKOS  
   Kokkos::deep_copy(adjacency_d_, adjacency_h_);
#endif
#ifdef USE_SYCL
	q_.memcpy(adjacency_d_, adjacency_,adjacencySize*sizeof(int));
	q_.wait();
#endif

   return;
}

/******************************************************************************/
void GeometryAAAB::sortFluidPts()
{

   int nInlet = 0;
   int nOutlet = 0;
   int nBorder = 0;
   for (int fluidIdx=0; fluidIdx<nFluid_; fluidIdx++)
   {
         if(checkPtOnTaskBoundary(fluidPts_[fluidIdx]))
         {
            nBorder++;
         }
      
   }

   startInlet_ = 0;
   countInlet_ = nInlet;
   startOutlet_ = countInlet_;
   countOutlet_ = nOutlet;
   startBorder_ = countInlet_ + countOutlet_;
   countBorder_ = nBorder;
   startBulk_ = countInlet_ + countOutlet_ + countBorder_;
   countBulk_ = nFluid_-startBulk_;

   sort(fluidPts_.begin(),fluidPts_.end(), [&](const vector<int> & lhs, const vector<int> & rhs)
   {
      int valLhs = _INVALID_;
      if(checkPtOnTaskBoundary(lhs)) { valLhs = 3; }
      if (valLhs == _INVALID_) { valLhs = 4; }

      int valRhs = _INVALID_;
      if(checkPtOnTaskBoundary(rhs)) { valRhs = 3; }
      if (valRhs == _INVALID_) { valRhs = 4; }

      return (valLhs < valRhs);
   });

   if (countInlet_ > 0)
   {
      sort(fluidPts_.begin()+startInlet_,fluidPts_.begin()+startInlet_+countInlet_, [&](const vector<int> & lhs, const vector<int> & rhs)
      {
         int valLhs = getSortIdxFromPt(lhs);
         int valRhs = getSortIdxFromPt(rhs);
         return (valLhs < valRhs);
      });
   }

   if (countOutlet_ > 0)
   {
      sort(fluidPts_.begin()+startOutlet_,fluidPts_.begin()+startOutlet_+countOutlet_, [&](const vector<int> & lhs, const vector<int> & rhs)
      {
         int valLhs = getSortIdxFromPt(lhs);
         int valRhs = getSortIdxFromPt(rhs);
         return (valLhs < valRhs);
      });
   }

   if (countBorder_ > 0)
   {
      sort(fluidPts_.begin()+startBorder_,fluidPts_.begin()+startBorder_+countBorder_, [&](const vector<int> & lhs, const vector<int> & rhs)
      {
         int valLhs = getSortIdxFromPt(lhs);
         int valRhs = getSortIdxFromPt(rhs);
         return (valLhs < valRhs);
      });
   }

   if (countBulk_ > 0)
   {
      sort(fluidPts_.begin()+startBulk_,fluidPts_.begin()+startBulk_+countBulk_, [&](const vector<int> & lhs, const vector<int> & rhs)
      {
         int valLhs = getSortIdxFromPt(lhs);
         int valRhs = getSortIdxFromPt(rhs);
         return (valLhs < valRhs);
      });
   }

   return;
}

/******************************************************************************/
bool GeometryAAAB::checkPtOnInlet(const vector<int>& fluidPt)
{

   bool inletPt = false;
   if (fluidPt[0] < lattice_.interactDistance)
   {
      inletPt = true;
   }
   return inletPt;
}

/******************************************************************************/
bool GeometryAAAB::checkPtOnOutlet(const vector<int>& fluidPt)
{

   bool outletPt = false;
   if (fluidPt[0] > totalLength_[0]-1-lattice_.interactDistance)
   {
      outletPt = true;
   }
   return outletPt;
}

/******************************************************************************/
bool GeometryAAAB::checkPtOnTaskBoundary(const vector<int>& fluidPt)
{

   bool boundaryPt = false;
   if      (fluidPt[0]-lattice_.interactDistance < myMin_[0] || fluidPt[0]+lattice_.interactDistance > myMax_[0]) { boundaryPt = true; }
   else if (fluidPt[1]-lattice_.interactDistance < myMin_[1] || fluidPt[1]+lattice_.interactDistance > myMax_[1]) { boundaryPt = true; }
   else if (fluidPt[2]-lattice_.interactDistance < myMin_[2] || fluidPt[2]+lattice_.interactDistance > myMax_[2]) { boundaryPt = true; }
   return boundaryPt;
}

/******************************************************************************/
int GeometryAAAB::getSortIdxFromPt(const vector<int>& fluidPt)
{
   return myDims_[2] * myDims_[1] * (fluidPt[0] - myMin_[0]) + myDims_[1] * (fluidPt[1] - myMin_[1]) + fluidPt[2] - myMin_[2];
}

/******************************************************************************/
