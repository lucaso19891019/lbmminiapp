
#include <algorithm>
#include <cassert>
#include <iostream>
#include <mpi.h>

#include "Communication.hh"
#include "Parameters.hh"

/******************************************************************************/
#if defined(USE_SYCL)
Communication::Communication(Lattice& lattice, Geometry& geometry, Parameters& parameters,sycl::queue& q):
   lattice_(lattice), geometry_(geometry), parameters_(parameters),q_(q)
#else
Communication::Communication(Lattice& lattice, Geometry& geometry, Parameters& parameters):
   lattice_(lattice), geometry_(geometry), parameters_(parameters)
#endif
{
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank_);
   MPI_Comm_size(MPI_COMM_WORLD, &nTasks_);

   interactDist_ = lattice_.interactDistance;

   commLength_ = 0;
   numNbrs_ = 0;
}


/******************************************************************************/
void Communication::identifyNeighbors()
{

   vector<int> minByTask(_DIMS_*nTasks_, 0);
   vector<int> maxByTask(_DIMS_*nTasks_, 0);

   minByTask[_DIMS_*myRank_ + 0] = geometry_.getMinByDim(0);
   minByTask[_DIMS_*myRank_ + 1] = geometry_.getMinByDim(1);
   minByTask[_DIMS_*myRank_ + 2] = geometry_.getMinByDim(2);
   maxByTask[_DIMS_*myRank_ + 0] = geometry_.getMaxByDim(0);
   maxByTask[_DIMS_*myRank_ + 1] = geometry_.getMaxByDim(1);
   maxByTask[_DIMS_*myRank_ + 2] = geometry_.getMaxByDim(2);

   MPI_Allreduce(MPI_IN_PLACE,&minByTask[0],3*nTasks_,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
   MPI_Allreduce(MPI_IN_PLACE,&maxByTask[0],3*nTasks_,MPI_INT,MPI_SUM,MPI_COMM_WORLD);


   for (int task=0; task<nTasks_; task++)
   {
      if (task != myRank_)
      {
         if (isTaskNbr(minByTask, maxByTask, task) || (isTaskNbrPeriodic(minByTask, maxByTask, task)))
         {
            neighbors_.push_back(task);
            vector<int> nbrMin = {minByTask[_DIMS_*task + 0],
                                  minByTask[_DIMS_*task + 1],
                                  minByTask[_DIMS_*task + 2]};
            vector<int> nbrMax = {maxByTask[_DIMS_*task + 0],
                                  maxByTask[_DIMS_*task + 1],
                                  maxByTask[_DIMS_*task + 2]};
            nbrMinByTask_.push_back(nbrMin);
            nbrMaxByTask_.push_back(nbrMax);
         }
      }
   }

   numNbrs_ = neighbors_.size();

   return;
}

/******************************************************************************/
void Communication::identifyPatternAB()
{

   int nFluid = geometry_.getNumFluidPts();

   vector<vector<int> > sendOrderByTask;
   sendOrderByTask.resize(numNbrs_);

   vector<vector<int> > locsToSend;
   locsToSend.resize(numNbrs_);

   for (int fluidIdx = geometry_.getInletStart(); fluidIdx < geometry_.getBorderStart()+geometry_.getBorderCount(); fluidIdx++)
   {
      int myI = geometry_.fluidPts_[fluidIdx][0];
      int myJ = geometry_.fluidPts_[fluidIdx][1];
      int myK = geometry_.fluidPts_[fluidIdx][2];
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
         int nbrI = myI + lattice_.getVel(stencilIdx, 0);
         int nbrJ = myJ + lattice_.getVel(stencilIdx, 1);
         int nbrK = myK + lattice_.getVel(stencilIdx, 2);
            if (nbrI < 0)
            {
               nbrI += geometry_.getTotalLengthByDim(0);
            }
            else if (nbrI >= geometry_.getTotalLengthByDim(0))
            {
               nbrI -= geometry_.getTotalLengthByDim(0);
            }
         
         for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
         {
            if (ptBelongsToNbr(nbrI, nbrJ, nbrK, nbrIdx))
            {
               sendOrderByTask[nbrIdx].push_back(nbrI);
               sendOrderByTask[nbrIdx].push_back(nbrJ);
               sendOrderByTask[nbrIdx].push_back(nbrK);
               sendOrderByTask[nbrIdx].push_back(stencilIdx);
               // store locations of location after bounceback
#ifdef SOA
               locsToSend[nbrIdx].push_back(nFluid * lattice_.getOpp(stencilIdx) + fluidIdx);
#elif AOS
               locsToSend[nbrIdx].push_back(_LATTICESIZE_ * fluidIdx + lattice_.getOpp(stencilIdx));
#endif
            }
         }
      }
   }

   vector<vector<int> > recvOrderByTask;
   recvOrderByTask.resize(numNbrs_);
   exchangeArrays(sendOrderByTask, recvOrderByTask);

   vector<vector<int> > sendRejectsByTask;
   sendRejectsByTask.resize(numNbrs_);

   vector<vector<int> > locsToRecv;
   locsToRecv.resize(numNbrs_);

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int numRecvPts = recvOrderByTask[nbrIdx].size()/4;
      assert(4*numRecvPts == recvOrderByTask[nbrIdx].size());
      for (int ptIdx = 0; ptIdx < numRecvPts; ptIdx++)
      {
         int myI = recvOrderByTask[nbrIdx][4*ptIdx+0];
         int myJ = recvOrderByTask[nbrIdx][4*ptIdx+1];
         int myK = recvOrderByTask[nbrIdx][4*ptIdx+2];
         int myStencilIdx = recvOrderByTask[nbrIdx][4*ptIdx+3];
         int myIJK = geometry_.getGridIdx(myI, myJ, myK);
         int myFluidIdx = geometry_.getGridElement(myIJK);
         if (myFluidIdx >= 0)
         {
#ifdef SOA
            locsToRecv[nbrIdx].push_back(nFluid * myStencilIdx + myFluidIdx);
#elif AOS
            locsToRecv[nbrIdx].push_back(_LATTICESIZE_ * myFluidIdx + myStencilIdx);
#endif
         }
         else
         {
            sendRejectsByTask[nbrIdx].push_back(ptIdx);
         }
      }
   }

   removeRejectedPts(sendRejectsByTask, locsToSend);
   setupBuffers(locsToSend, locsToRecv);

   return;
}

/******************************************************************************/
void Communication::identifyPatternAA()
{

   int nFluid = geometry_.getNumFluidPts();

   vector<vector<int> > sendOrderByTask;
   sendOrderByTask.resize(numNbrs_);

   vector<vector<int> > locsToSend;
   locsToSend.resize(numNbrs_);

   for (int fluidIdx = geometry_.getInletStart(); fluidIdx < geometry_.getBorderStart()+geometry_.getBorderCount(); fluidIdx++)
   {
      int myI = geometry_.fluidPts_[fluidIdx][0];
      int myJ = geometry_.fluidPts_[fluidIdx][1];
      int myK = geometry_.fluidPts_[fluidIdx][2];
      for (int stencilIdx = 0; stencilIdx < _LATTICESIZE_; stencilIdx++)
      {
         int nbrI = myI + lattice_.getVel(stencilIdx, 0);
         int nbrJ = myJ + lattice_.getVel(stencilIdx, 1);
         int nbrK = myK + lattice_.getVel(stencilIdx, 2);
            if (nbrI < 0)
            {
               nbrI += geometry_.getTotalLengthByDim(0);
            }
            else if (nbrI >= geometry_.getTotalLengthByDim(0))
            {
               nbrI -= geometry_.getTotalLengthByDim(0);
            }
         
         for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
         {
            if (ptBelongsToNbr(nbrI, nbrJ, nbrK, nbrIdx))
            {
               sendOrderByTask[nbrIdx].push_back(nbrI);
               sendOrderByTask[nbrIdx].push_back(nbrJ);
               sendOrderByTask[nbrIdx].push_back(nbrK);
               sendOrderByTask[nbrIdx].push_back(stencilIdx);
               // store locations of location after bounceback
#ifdef SOA
               locsToSend[nbrIdx].push_back(nFluid * stencilIdx + fluidIdx);
#elif AOS
               locsToSend[nbrIdx].push_back(_LATTICESIZE_ * fluidIdx + stencilIdx);
#endif
            }
         }
      }
   }

   vector<vector<int> > recvOrderByTask;
   recvOrderByTask.resize(numNbrs_);
   exchangeArrays(sendOrderByTask, recvOrderByTask);

   vector<vector<int> > sendRejectsByTask;
   sendRejectsByTask.resize(numNbrs_);

   vector<vector<int> > locsToRecv;
   locsToRecv.resize(numNbrs_);

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int numRecvPts = recvOrderByTask[nbrIdx].size()/4;
      assert(4*numRecvPts == recvOrderByTask[nbrIdx].size());
      for (int ptIdx = 0; ptIdx < numRecvPts; ptIdx++)
      {
         int myI = recvOrderByTask[nbrIdx][4*ptIdx+0];
         int myJ = recvOrderByTask[nbrIdx][4*ptIdx+1];
         int myK = recvOrderByTask[nbrIdx][4*ptIdx+2];
         int myStencilIdx = recvOrderByTask[nbrIdx][4*ptIdx+3];
         int myIJK = geometry_.getGridIdx(myI, myJ, myK);
         int myFluidIdx = geometry_.getGridElement(myIJK);
         if (myFluidIdx >= 0)
         {
#ifdef SOA
            locsToRecv[nbrIdx].push_back(nFluid * lattice_.getOpp(myStencilIdx) + myFluidIdx);
#elif AOS
            locsToRecv[nbrIdx].push_back(_LATTICESIZE_ * myFluidIdx + lattice_.getOpp(myStencilIdx));
#endif
         }
         else
         {
            sendRejectsByTask[nbrIdx].push_back(ptIdx);
         }
      }
   }

   removeRejectedPts(sendRejectsByTask, locsToSend);
   setupBuffers(locsToSend, locsToRecv);

   return;
}

/******************************************************************************/
void Communication::setupBuffers(vector<vector<int> >& locsToSend, vector<vector<int> >& locsToRecv)
{

   int commLength = 0;
   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      assert(locsToSend[nbrIdx].size() == locsToRecv[nbrIdx].size());
      commLength += locsToSend[nbrIdx].size();
   }
   commLength_ = commLength;
#if defined(USE_KOKKOS)
   locsToSend_d_ = myViewInt("locsToSend_d_",commLength);
   locsToRecv_d_ = myViewInt("locsToRecv_d_",commLength);
   locsToSend_h_ = Kokkos::create_mirror_view(locsToSend_d_);
   locsToRecv_h_ = Kokkos::create_mirror_view(locsToRecv_d_);
   int counterComm = 0;
   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      for (int ii = 0; ii < locsToSend[nbrIdx].size(); ii++)
      {
         locsToSend_h_(counterComm) = locsToSend[nbrIdx][ii];
         locsToRecv_h_(counterComm) = locsToRecv[nbrIdx][ii];
         counterComm++;
      }
   }
   Kokkos::deep_copy(locsToSend_d_, locsToSend_h_);
   Kokkos::deep_copy(locsToRecv_d_, locsToRecv_h_);
#else
   locsToSend_ = new int[commLength_];
   locsToRecv_ = new int[commLength_];
   fill(locsToSend_, locsToSend_+commLength_, _INVALID_);
   fill(locsToRecv_, locsToRecv_+commLength_, _INVALID_);

   int counterComm = 0;
   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      for (int ii = 0; ii < locsToSend[nbrIdx].size(); ii++)
      {
         locsToSend_[counterComm] = locsToSend[nbrIdx][ii];
         locsToRecv_[counterComm] = locsToRecv[nbrIdx][ii];
         counterComm++;
      }
   }
#if defined(USE_SYCL)
    locsToSend_d_=sycl::malloc_device<int>(commLength_,q_);
    locsToRecv_d_=sycl::malloc_device<int>(commLength_,q_);
    q_.memcpy(locsToSend_d_,locsToSend_,commLength_*sizeof(int));
    q_.memcpy(locsToRecv_d_,locsToRecv_,commLength_*sizeof(int));
    q_.wait();
#endif
#endif

#if defined(USE_KOKKOS)
   bufferSend_ = myViewPDF("bufferSend_",commLength_);
   bufferRecv_ = myViewPDF("bufferRecv_",commLength_);
#elif defined(USE_SYCL)
   bufferSend_=sycl::malloc_device<Pdf>(commLength_,q_);
   bufferRecv_=sycl::malloc_device<Pdf>(commLength_,q_);
#else
   bufferSend_ = new Pdf[commLength_];
   bufferRecv_ = new Pdf[commLength_];
   fill(bufferSend_, bufferSend_+commLength_, 0.0);
   fill(bufferRecv_, bufferRecv_+commLength_, 0.0);
#endif

   bufferStartByNbr_.resize(numNbrs_);
   bufferCountByNbr_.resize(numNbrs_);
   int runningCount = 0;
   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      bufferStartByNbr_[nbrIdx] = runningCount;
      bufferCountByNbr_[nbrIdx] = locsToSend[nbrIdx].size();
      runningCount += bufferCountByNbr_[nbrIdx];
   }

   return;
}
/******************************************************************************/
void Communication::kokkosSyclNoNeighborSetup()
{
#if defined(USE_SYCL)
   locsToSend_d_=sycl::malloc_device<int>(commLength_,q_);
   locsToRecv_d_=sycl::malloc_device<int>(commLength_,q_);
   bufferSend_=sycl::malloc_device<Pdf>(commLength_,q_);
   bufferRecv_=sycl::malloc_device<Pdf>(commLength_,q_);
#endif
#if defined(USE_KOKKOS)
   locsToSend_d_ = myViewInt("locsToSend_d_",commLength);
   locsToRecv_d_ = myViewInt("locsToRecv_d_",commLength);
   bufferSend_ = myViewPDF("bufferSend_",commLength_);
   bufferRecv_ = myViewPDF("bufferRecv_",commLength_);
#endif
}


/******************************************************************************/
void Communication::removeRejectedPts(vector<vector<int> >& sendRejectsByTask, vector<vector<int> >& locsToSend)
{

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      sort(sendRejectsByTask[nbrIdx].rbegin(), sendRejectsByTask[nbrIdx].rend());
   }
   vector<vector<int> > recvRejectsByTask;
   recvRejectsByTask.resize(numNbrs_);
   exchangeArrays(sendRejectsByTask, recvRejectsByTask);

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int numRejectPts = recvRejectsByTask[nbrIdx].size();
      for (int ptIdx = 0; ptIdx < numRejectPts; ptIdx++)
      {
         int rejectIdx = recvRejectsByTask[nbrIdx][ptIdx];
         locsToSend[nbrIdx].erase(locsToSend[nbrIdx].begin()+rejectIdx);
      }
   }

   return;
}

/******************************************************************************/
void Communication::exchange(Pdf* dstrb)

{
#ifdef USE_KOKKOS
   
   Kokkos::parallel_for(myPolicy(0, commLength_),KOKKOS_CLASS_LAMBDA (const int commIdx)
   {
   	  int dataLoc = locsToSend_d_(commIdx);
   	  bufferSend_(commIdx) = dstrb[dataLoc];
   });

   Kokkos::fence();
   
   MPI_Request* sendRequest = new MPI_Request[numNbrs_];
   MPI_Request* recvRequest = new MPI_Request[numNbrs_];
   int tag = 103;

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      int bufferStart = bufferStartByNbr_[nbrIdx];
      int bufferCount = bufferCountByNbr_[nbrIdx];
      MPI_Irecv(bufferRecv_.data()+bufferStart, bufferCount, MPI_DOUBLE, nbrRank, tag, MPI_COMM_WORLD, recvRequest+nbrIdx);
   }

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      int bufferStart = bufferStartByNbr_[nbrIdx];
      int bufferCount = bufferCountByNbr_[nbrIdx];   
      MPI_Isend(bufferSend_.data()+bufferStart, bufferCount, MPI_DOUBLE, nbrRank, tag, MPI_COMM_WORLD, sendRequest+nbrIdx);
   }

   MPI_Waitall(numNbrs_, sendRequest, MPI_STATUSES_IGNORE);
   MPI_Waitall(numNbrs_, recvRequest, MPI_STATUSES_IGNORE);
   delete[] sendRequest;
   delete[] recvRequest;
   
   Kokkos::fence();
   Kokkos::parallel_for(myPolicy(0, commLength_),KOKKOS_CLASS_LAMBDA (const int commIdx)
   {
   	  int dataLoc = locsToRecv_d_(commIdx);
   	  dstrb[dataLoc] = bufferRecv_(commIdx);
   });

#else
#if defined(USE_SYCL)
	q_.parallel_for(sycl::range<1>{size_t(commLength_)},[=,locsToSend_d_=this->locsToSend_d_,bufferSend_=this->bufferSend_](sycl::id<1> commIdx){
	int dataLoc = locsToSend_d_[commIdx];
      bufferSend_[commIdx] = dstrb[dataLoc];
	});
	q_.wait();
#else
   for (int commIdx = 0; commIdx < commLength_; commIdx++)
   {
      int dataLoc = locsToSend_[commIdx];
      bufferSend_[commIdx] = dstrb[dataLoc];
   }
#endif

   MPI_Request* sendRequest = new MPI_Request[numNbrs_];
   MPI_Request* recvRequest = new MPI_Request[numNbrs_];
   int tag = 103;

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      int bufferStart = bufferStartByNbr_[nbrIdx];
      int bufferCount = bufferCountByNbr_[nbrIdx];
      MPI_Irecv(&bufferRecv_[bufferStart], bufferCount, MPI_DOUBLE, nbrRank, tag, MPI_COMM_WORLD, recvRequest+nbrIdx);
   }

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      int bufferStart = bufferStartByNbr_[nbrIdx];
      int bufferCount = bufferCountByNbr_[nbrIdx];
      MPI_Isend(&bufferSend_[bufferStart], bufferCount, MPI_DOUBLE, nbrRank, tag, MPI_COMM_WORLD, sendRequest+nbrIdx);
   }

   MPI_Waitall(numNbrs_, sendRequest, MPI_STATUSES_IGNORE);
   MPI_Waitall(numNbrs_, recvRequest, MPI_STATUSES_IGNORE);
   delete[] sendRequest;
   delete[] recvRequest;
#if defined(USE_SYCL)
	q_.wait();
	q_.parallel_for(sycl::range<1>{size_t(commLength_)},[=,locsToRecv_d_=this->locsToRecv_d_,bufferRecv_=this->bufferRecv_](sycl::id<1> commIdx){
	int dataLoc = locsToRecv_d_[commIdx];
      dstrb[dataLoc] = bufferRecv_[commIdx];
	});
	
#else
   for (int commIdx = 0; commIdx < commLength_; commIdx++)
   {
      int dataLoc = locsToRecv_[commIdx];
      dstrb[dataLoc] = bufferRecv_[commIdx];
   }
#endif   
#endif

   return;

}

/******************************************************************************/
/******************************************************************************/
bool Communication::ptBelongsToNbr(int i, int j, int k, int nbrIdx)
{
   if (nbrMinByTask_[nbrIdx][0] > i) return false;
   if (nbrMinByTask_[nbrIdx][1] > j) return false;
   if (nbrMinByTask_[nbrIdx][2] > k) return false;
   if (nbrMaxByTask_[nbrIdx][0] < i) return false;
   if (nbrMaxByTask_[nbrIdx][1] < j) return false;
   if (nbrMaxByTask_[nbrIdx][2] < k) return false;
   return true;
}

/******************************************************************************/
void Communication::exchangeArraySizes(vector<vector<int> >& arraySend, vector<int>& sizeRecv)
{

   MPI_Request* sendRequest = new MPI_Request[numNbrs_];
   MPI_Request* recvRequest = new MPI_Request[numNbrs_];
   int tag = 100;

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      MPI_Irecv(&sizeRecv[nbrIdx], 1, MPI_INT, nbrRank, tag, MPI_COMM_WORLD, recvRequest+nbrIdx);
   }
   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      int sizeSend = arraySend[nbrIdx].size();
      MPI_Isend(&sizeSend, 1, MPI_INT, nbrRank, tag, MPI_COMM_WORLD, sendRequest+nbrIdx);
   }

   MPI_Waitall(numNbrs_, sendRequest, MPI_STATUSES_IGNORE);
   MPI_Waitall(numNbrs_, recvRequest, MPI_STATUSES_IGNORE);
   delete[] sendRequest;
   delete[] recvRequest;
   return;
}

/******************************************************************************/
void Communication::exchangeArrays(vector<vector<int> >& arraySend, vector<vector<int> >& arrayRecv)
{

   vector<int> sizeRecv(numNbrs_);
   exchangeArraySizes(arraySend, sizeRecv);
   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      arrayRecv[nbrIdx].resize(sizeRecv[nbrIdx]);
   }

   MPI_Request* sendRequest = new MPI_Request[numNbrs_];
   MPI_Request* recvRequest = new MPI_Request[numNbrs_];
   int tag = 101;

   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      MPI_Irecv(&arrayRecv[nbrIdx][0], sizeRecv[nbrIdx], MPI_INT, nbrRank, tag, MPI_COMM_WORLD, recvRequest+nbrIdx);
   }
   for (int nbrIdx = 0; nbrIdx < numNbrs_; nbrIdx++)
   {
      int nbrRank = neighbors_[nbrIdx];
      int sizeSend = arraySend[nbrIdx].size();
      MPI_Isend(&arraySend[nbrIdx][0], sizeSend, MPI_INT, nbrRank, tag, MPI_COMM_WORLD, sendRequest+nbrIdx);
   }

   MPI_Waitall(numNbrs_, sendRequest, MPI_STATUSES_IGNORE);
   MPI_Waitall(numNbrs_, recvRequest, MPI_STATUSES_IGNORE);
   delete[] sendRequest;
   delete[] recvRequest;
   return;

}

/******************************************************************************/
bool Communication::isTaskNbr(vector<int> minByTask,  vector<int> maxByTask, int nbrTask)
{

   if (minByTask[_DIMS_*nbrTask + 0] - interactDist_ > geometry_.getMaxByDim(0)) { return false; }
   if (minByTask[_DIMS_*nbrTask + 1] - interactDist_ > geometry_.getMaxByDim(1)) { return false; }
   if (minByTask[_DIMS_*nbrTask + 2] - interactDist_ > geometry_.getMaxByDim(2)) { return false; }
   if (maxByTask[_DIMS_*nbrTask + 0] + interactDist_ < geometry_.getMinByDim(0)) { return false; }
   if (maxByTask[_DIMS_*nbrTask + 1] + interactDist_ < geometry_.getMinByDim(1)) { return false; }
   if (maxByTask[_DIMS_*nbrTask + 2] + interactDist_ < geometry_.getMinByDim(2)) { return false; }
   return true;
}

/******************************************************************************/
bool Communication::isTaskNbrPeriodic(vector<int> minByTask,  vector<int> maxByTask, int nbrTask)
{

   if (minByTask[_DIMS_*nbrTask + 0] - interactDist_ < 0 && geometry_.getMaxByDim(0) + interactDist_ >= geometry_.getTotalLengthByDim(0)) { return true; }
   if (maxByTask[_DIMS_*nbrTask + 0] + interactDist_ >= geometry_.getTotalLengthByDim(0) && geometry_.getMinByDim(0) - interactDist_ < 0) { return true; }
   return false;
}

/******************************************************************************/
