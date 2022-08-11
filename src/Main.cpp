

#include <iostream>
#include <mpi.h>



#include "Communication.hh"
#include "Geometry.hh"
#include "GeometryAAAB.hh"
#include "Lattice.hh"
#include "LatticeD3Q19.hh"
#include "LatticeD3Q27.hh"
#include "Parameters.hh"
#include "Propagation.hh"
#include "PropagationAA.hh"
#include "PropagationAB.hh"


using namespace::std;

int main(int argc, char** argv)
{

   MPI_Init(&argc, &argv);
   int myRank, nTasks;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
   MPI_Comm_size(MPI_COMM_WORLD, &nTasks);

#if defined(USE_KOKKOS)
	Kokkos::initialize(argc, argv);
   {
#elif defined(USE_SYCL)
	sycl::queue q(sycl::default_selector{});
	{
#endif

   if (myRank == 0)
   {
      cout << "Run started on " << nTasks << " tasks" << endl;
   }

   double tStart = MPI_Wtime();

   // setup basic parameters
   Parameters parameters;
   string inputFile(argv[1]);
   parameters.readInput(inputFile);

   // setup lattice
   Lattice* lattice = NULL;
   if (_LATTICESIZE_ == 19)
   {
      lattice = new LatticeD3Q19;
   }
   else if (_LATTICESIZE_ == 27)
   {
      lattice = new LatticeD3Q27;
   }

   // setup grid
   // Geometry geometry(*lattice, parameters);
   Geometry* geometry = NULL;
   int propPattern = parameters.getPropPattern();
#if defined(USE_SYCL)
	geometry = new GeometryAAAB(*lattice, parameters,q);
#else
   geometry = new GeometryAAAB(*lattice, parameters);
#endif
   geometry->decomposeDomain();
   geometry->setupGrid();
   geometry->setup();
   geometry->assignFluidIndices();
   geometry->setupAdjacency();


   // setup communication
#if defined(USE_SYCL)
	Communication comm(*lattice, *geometry, parameters,q);
#else
   Communication comm(*lattice, *geometry, parameters);
#endif
   ;
   // setup propagation
   Propagation* prop = NULL;
   if (propPattern == 0)
   {
      if (nTasks > 1)
      {
         comm.identifyNeighbors();
         comm.identifyPatternAB();
      }
      else
      {
	 comm.kokkosSyclNoNeighborSetup();
      }
#if defined(USE_SYCL)
	prop = new PropagationAB(*lattice, *geometry, comm, parameters,q);
#else
      prop = new PropagationAB(*lattice, *geometry, comm, parameters);
#endif

   }
   else if (propPattern == 1)
   {
      if (nTasks > 1)
      {
         comm.identifyNeighbors();
         comm.identifyPatternAA();
      }
      else
      {
	 comm.kokkosSyclNoNeighborSetup();
      }

#if defined(USE_SYCL)
	prop = new PropagationAA(*lattice, *geometry, comm, parameters,q);
#else
      prop = new PropagationAA(*lattice, *geometry, comm, parameters);
#endif

   }

   // run propagation
   prop->setup();
   prop->run();

   MPI_Barrier(MPI_COMM_WORLD);
   double tEnd = MPI_Wtime();
   if (myRank == 0)
   {
      cout << "Total runtime (rank 0): " << tEnd-tStart << endl;
   }

#if defined(USE_KOKKOS)
   }
	Kokkos::finalize();
#elif defined(USE_SYCL)
	}
#endif

   MPI_Finalize();

   return 0;
}
