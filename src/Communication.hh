#ifndef COMMUNICATION_H
#define COMMUNICATION_H


#include "Geometry.hh"
#include "Lattice.hh"
#include "Parameters.hh"


class Communication
{
   public:
#if defined(USE_SYCL)
      Communication(Lattice& lattice, Geometry& geometry, Parameters& parameters,sycl::queue& q);
#else
	Communication(Lattice& lattice, Geometry& geometry, Parameters& parameters);
#endif
      ~Communication(void) {
#if defined(USE_SYCL)
	sycl::free(bufferSend_,q_);
	sycl::free(bufferRecv_,q_);
	sycl::free(locsToSend_d_,q_);
	sycl::free(locsToRecv_d_,q_);
#endif
};

      void identifyNeighbors(void);
      void identifyPatternAB(void);
      void identifyPatternAA(void);
      void kokkosSyclNoNeighborSetup(void);
      void exchange(Pdf* dstrb);


      int getCommLength(void) { return commLength_; }

      int *locsToSend_, *locsToRecv_;
#if defined(USE_KOKKOS)
      myViewInt locsToSend_d_, locsToRecv_d_;
      myMirrorViewInt locsToSend_h_, locsToRecv_h_;
#else
      int *locsToSend_d_, *locsToRecv_d_;
#endif

   protected:

      Lattice& lattice_;
      Geometry& geometry_;
      Parameters& parameters_;
#if defined(USE_SYCL)
      sycl::queue& q_;
#endif

   private:

      void setupBuffers(vector<vector<int> >& locsToSend, vector<vector<int> >& locsToRecv);
      void removeRejectedPts(vector<vector<int> >& sendRejectsByTask, vector<vector<int> >& locsToSend);
      bool ptBelongsToNbr(int i, int j, int k, int nbrIdx);
      void exchangeArraySizes(vector<vector<int> >& arraySend, vector<int>& sizeRecv);
      void exchangeArrays(vector<vector<int> >& arraySend, vector<vector<int> >& arrayRecv);
      bool isTaskNbr(vector<int> minByTask,  vector<int> maxByTask, int nbrTask);
      bool isTaskNbrPeriodic(vector<int> minByTask,  vector<int> maxByTask, int nbrTask);

      int myRank_, nTasks_;
      int interactDist_;
      int numNbrs_, commLength_;

      vector<int> neighbors_;
      vector<vector<int> > nbrMinByTask_, nbrMaxByTask_;

#ifdef USE_KOKKOS
      myViewPDF bufferSend_, bufferRecv_;
#else
      Pdf *bufferSend_, *bufferRecv_;
#endif
      vector<int> bufferStartByNbr_, bufferCountByNbr_;

};
#endif
