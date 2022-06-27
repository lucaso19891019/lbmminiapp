#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>

#if defined(USE_KOKKOS_CUDA)
#include <Kokkos_Core.hpp>
#define myDeviceMemorySpace Kokkos::CudaSpace
#define myPolicy Kokkos::RangePolicy<Kokkos::Cuda::execution_space>
#define USE_KOKKOS
#elif defined(USE_KOKKOS_SYCL)
#include <Kokkos_Core.hpp>
#define myDeviceMemorySpace Kokkos::Experimental::SYCLDeviceUSMSpace
#define myPolicy Kokkos::RangePolicy<myDeviceMemorySpace::execution_space>
#define USE_KOKKOS
#elif defined(USE_KOKKOS_OPENMPTARGET)
#include <Kokkos_Core.hpp>
#define myDeviceMemorySpace Kokkos::Experimental::OpenMPTargetSpace
#define myPolicy Kokkos::RangePolicy<myDeviceMemorySpace::execution_space>
#define USE_KOKKOS
#endif

#ifdef USE_KOKKOS
#define myViewInt Kokkos::View<int *, myDeviceMemorySpace>
#define myMirrorViewInt Kokkos::View<int *, myDeviceMemorySpace>::HostMirror

#define myViewPDF Kokkos::View<Pdf *, myDeviceMemorySpace>
#define myMirrorViewPDF Kokkos::View<Pdf *, myDeviceMemorySpace>::HostMirror
#endif

#ifdef USE_SYCL
#include <CL/sycl.hpp>
#endif

#include "Lattice.hh"
#include "Parameters.hh"

using namespace::std;

enum LAT_PT_TYPES_ {
   LAT_PT_WALL = -10,   // Generic description of non-fluid points
   LAT_PT_INLET = -11,  // Fluid points at inlet
   LAT_PT_OUTLET = -12  // Fluid points at outlet
};

class Geometry
{
   public:
#ifdef USE_SYCL
      Geometry(Lattice& lattice, Parameters& parameters, sycl::queue& q);
#else
	Geometry(Lattice& lattice, Parameters& parameters);
#endif
      virtual ~Geometry(void) {
#ifdef USE_SYCL
	sycl::free(adjacency_d_,q_);
#endif
};
      void decomposeDomain();
      void setupGrid();
      void assignFluidIndices();
      virtual void setup() = 0;
      virtual void setupAdjacency() = 0;

      int getGridIdx(int i, int j, int k);
      int64_t getGlobalNumFluidPts() {return nFluidGlobal_; }
      int getNumFluidPts() {return nFluid_; }
      int getTotalLengthByDim(int dim) {return totalLength_[dim]; }
      int getMinByDim(int dim) {return myMin_[dim]; }
      int getMaxByDim(int dim) {return myMax_[dim]; }
      int getDimsByDim(int dim) {return myDims_[dim]; }
      int getGridElement(int idx) {return grid_[idx]; }

      int getInletStart() { return startInlet_; }
      int getInletCount() { return countInlet_; }
      int getOutletStart() { return startOutlet_; }
      int getOutletCount() { return countOutlet_; }
      int getBorderStart() { return startBorder_; }
      int getBorderCount() { return countBorder_; }
      int getBulkStart() { return startBulk_; }
      int getBulkCount() { return countBulk_; }

      int getNumLayers() { return numLayers_; }
      int getNumTempLayers() {return numTempLayers_; }
      int getStartByLayer(int layer) {return startByLayer_[layer]; }
      int getCountByLayer(int layer) {return countByLayer_[layer]; }
      int getMaxPtsLayer() {return maxPtsLayer_; }
      int getTempPosition(int stencilIdx, int relLayer, int layerPosition)
      {
         return stencilIdx + _LATTICESIZE_ * layerPosition + _LATTICESIZE_ * maxPtsLayer_ * relLayer;
      }

#ifdef USE_KOKKOS
      myViewInt adjacency_d_;
      myMirrorViewInt adjacency_h_;
#else
      int *adjacency_d_;
#endif
      int *adjacency_;
      int *borderMap_, *borderMap_d_;
      int* borderMap2_;
      vector<vector<int> > fluidPts_;
      vector<int> inletPts_, outletPts_;
      vector<vector<int> > fdInletPts_, fdOutletPts_;

   protected:
      int myRank_, nTasks_;

      Lattice& lattice_;
      Parameters& parameters_;
#if defined(USE_SYCL)
      sycl::queue& q_;
#endif
      vector<int> boxes_, totalLength_;
      vector<int> myMin_, myMax_, myDims_;

      vector<int> grid_;

      int64_t nFluidGlobal_;
      int nFluid_;
      int startInlet_, countInlet_, startOutlet_, countOutlet_;
      int startBorder_, countBorder_, startBulk_, countBulk_;

      int numLayers_, numTempLayers_, maxPtsLayer_, voidMoment_, voidDstrb_;
      vector<int> startByLayer_, countByLayer_;

};
#endif
