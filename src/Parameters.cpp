
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <mpi.h>
#include <math.h>
#include <vector>

#include "Parameters.hh"

/******************************************************************************/
Parameters::Parameters()
{
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank_);
   MPI_Comm_size(MPI_COMM_WORLD, &nTasks_);
}

/******************************************************************************/
void Parameters::readInput(string inputFile)
{

   densityChange_ = 0.01;
   geometryLibID_ = 1;     // hardcoded to cylinder for now
   propPattern_ = 0;       // AB = 0, AA = 1
   pullSetting_ = 0;       // No = 0; Yes = 1
   unrollSetting_ = 0;     // No = 0; Yes = 1
   simulationSize_ = 1.0;  // default: laptop-sized
   timesteps_ = 1;         // default: single timestep
   writeVtk_ = 0;          // default: do not write vtk
  
   if (myRank_ == 0)
   {
      ifstream readFile;
      readFile.open(inputFile.c_str(),ifstream::in);
      string line;
      while(getline(readFile, line))
      {
         cout << line << endl;
         istringstream iss(line);
         vector<string> lineParts;
         string temp;
         while (iss)
         {
            iss >> temp;
            lineParts.push_back(temp);
         }
         if (lineParts.size() > 0)
         {
            if (lineParts[0] == "densitychange" || lineParts[0] == "densityChange")
            {
               densityChange_ = atof(lineParts[1].c_str());
            }
            else if (lineParts[0] == "proppattern" || lineParts[0] == "propPattern")
            {
               propPattern_ = atoi(lineParts[1].c_str());
            }
            else if (lineParts[0] == "pull")
            {
               pullSetting_ = atoi(lineParts[1].c_str());
            }
            else if (lineParts[0] == "unroll")
            {
               unrollSetting_ = atoi(lineParts[1].c_str());
            }
            else if (lineParts[0] == "simulationsize" || lineParts[0] == "simulationSize")
            {
               simulationSize_ = atof(lineParts[1].c_str());
            }
            else if (lineParts[0] == "timesteps")
            {
               timesteps_ = atoi(lineParts[1].c_str());
            }
            else if (lineParts[0] == "writevtk")
            {
               writeVtk_ = atoi(lineParts[1].c_str());
            }
         }
      }
      readFile.close();
   }

   MPI_Bcast(&densityChange_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&simulationSize_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&propPattern_, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&pullSetting_, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&unrollSetting_, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&timesteps_, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&writeVtk_, 1, MPI_INT, 0, MPI_COMM_WORLD);

   return;
}

/******************************************************************************/
