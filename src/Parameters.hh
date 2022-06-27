#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>

#ifndef _LATTICESIZE_
#define _LATTICESIZE_ 19
#endif

#ifndef _INVCS2_
#define _INVCS2_ 3.0
#endif

#ifndef _MOMENTSIZE_
#define _MOMENTSIZE_ 10
#endif

#ifndef _INVALID_
#define _INVALID_ -1
#endif

#ifndef _DIMS_
#define _DIMS_ 3
#endif

#ifndef _OMEGA_
#define _OMEGA_ 1.0
#endif

#define _19W0_ 0.0555555555555555555555
#define _19W1_ 0.0277777777777777777777
#define _19W2_ 0.3333333333333333333333

#define _ONETHIRD_ 0.333333333333333333
#define _ONESIXTH_ 0.166666666666666666
#define _GRAVITY_ 0.000001

#ifdef _PRECISION_SINGLE_
   typedef float Pdf;
#else
   typedef double Pdf;
#endif

using namespace::std;

class Parameters
{
   public:
      Parameters(void);
      ~Parameters(void) {};
      void readInput(string inputFile);

     
      double getDensityChange() { return densityChange_; }
      int getGeometryLibID() {return geometryLibID_; }
      int getPropPattern() { return propPattern_; }
      int getPullSetting() { return pullSetting_; }
      int getUnrollSetting() { return unrollSetting_; }
      double getSimulationSize() { return simulationSize_; }
      int getTimesteps() { return timesteps_; }
      int getWriteVtk() { return writeVtk_; }

   private:
      int myRank_, nTasks_;

      double densityChange_, simulationSize_;
      int geometryLibID_, propPattern_, pullSetting_;
      int unrollSetting_, timesteps_, writeVtk_;

};
#endif
