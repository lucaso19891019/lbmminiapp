
#include "LatticeD3Q27.hh"

/******************************************************************************/
LatticeD3Q27::LatticeD3Q27()
{
   stencil_.resize(27);
   inv_cs2 = 3.0;
   cs2 = 1.0/inv_cs2;
   interactDistance = 1;

   stencil_[0].opp = 1;
   stencil_[0].dir[0] = 1;  stencil_[0].dir[1] = 0;  stencil_[0].dir[2] = 0;
   stencil_[0].c = 2.0/27.0;

   stencil_[1].opp = 0;
   stencil_[1].dir[0] = -1;  stencil_[1].dir[1] = 0;  stencil_[1].dir[2] = 0;
   stencil_[1].c = 2.0/27.0;

   stencil_[2].opp = 3;
   stencil_[2].dir[0] = 0;  stencil_[2].dir[1] = 1;  stencil_[2].dir[2] = 0;
   stencil_[2].c = 2.0/27.0;

   stencil_[3].opp = 2;
   stencil_[3].dir[0] = 0;  stencil_[3].dir[1] = -1;  stencil_[3].dir[2] = 0;
   stencil_[3].c = 2.0/27.0;

   stencil_[4].opp = 5;
   stencil_[4].dir[0] = 0;  stencil_[4].dir[1] = 0;  stencil_[4].dir[2] = 1;
   stencil_[4].c = 2.0/27.0;

   stencil_[5].opp = 4;
   stencil_[5].dir[0] = 0;  stencil_[5].dir[1] = 0;  stencil_[5].dir[2] = -1;
   stencil_[5].c = 2.0/27.0;

   stencil_[6].opp = 11;
   stencil_[6].dir[0] = 1;  stencil_[6].dir[1] = 1;  stencil_[6].dir[2] = 0;
   stencil_[6].c = 1.0/54.0;

   stencil_[7].opp = 10;
   stencil_[7].dir[0] = 1;  stencil_[7].dir[1] = -1;  stencil_[7].dir[2] = 0;
   stencil_[7].c = 1.0/54.0;

   stencil_[8].opp = 13;
   stencil_[8].dir[0] = 1;  stencil_[8].dir[1] = 0;  stencil_[8].dir[2] = 1;
   stencil_[8].c = 1.0/54.0;

   stencil_[9].opp = 12;
   stencil_[9].dir[0] = 1;  stencil_[9].dir[1] = 0;  stencil_[9].dir[2] = -1;
   stencil_[9].c = 1.0/54.0;

   stencil_[10].opp = 7;
   stencil_[10].dir[0] = -1;  stencil_[10].dir[1] = 1;  stencil_[10].dir[2] = 0;
   stencil_[10].c = 1.0/54.0;

   stencil_[11].opp = 6;
   stencil_[11].dir[0] = -1;  stencil_[11].dir[1] = -1;  stencil_[11].dir[2] = 0;
   stencil_[11].c = 1.0/54.0;

   stencil_[12].opp = 9;
   stencil_[12].dir[0] = -1;  stencil_[12].dir[1] = 0;  stencil_[12].dir[2] = 1;
   stencil_[12].c = 1.0/54.0;

   stencil_[13].opp = 8;
   stencil_[13].dir[0] = -1;  stencil_[13].dir[1] = 0;  stencil_[13].dir[2] = -1;
   stencil_[13].c = 1.0/54.0;

   stencil_[14].opp = 17;
   stencil_[14].dir[0] = 0;  stencil_[14].dir[1] = 1;  stencil_[14].dir[2] = 1;
   stencil_[14].c = 1.0/54.0;

   stencil_[15].opp = 16;
   stencil_[15].dir[0] = 0;  stencil_[15].dir[1] = 1;  stencil_[15].dir[2] = -1;
   stencil_[15].c = 1.0/54.0;

   stencil_[16].opp = 15;
   stencil_[16].dir[0] = 0;  stencil_[16].dir[1] = -1;  stencil_[16].dir[2] = 1;
   stencil_[16].c = 1.0/54.0;

   stencil_[17].opp = 14;
   stencil_[17].dir[0] = 0;  stencil_[17].dir[1] = -1;  stencil_[17].dir[2] = -1;
   stencil_[17].c = 1.0/54.0;

   stencil_[18].opp = 25;
   stencil_[18].dir[0] = 1;  stencil_[18].dir[1] = 1;  stencil_[18].dir[2] = 1;
   stencil_[18].c = 1.0/216.0;

   stencil_[19].opp = 24;
   stencil_[19].dir[0] = 1;  stencil_[19].dir[1] = 1;  stencil_[19].dir[2] = -1;
   stencil_[19].c = 1.0/216.0;

   stencil_[20].opp = 23;
   stencil_[20].dir[0] = 1;  stencil_[20].dir[1] = -1;  stencil_[20].dir[2] = 1;
   stencil_[20].c = 1.0/216.0;

   stencil_[21].opp = 22;
   stencil_[21].dir[0] = 1;  stencil_[21].dir[1] = -1;  stencil_[21].dir[2] = -1;
   stencil_[21].c = 1.0/216.0;

   stencil_[22].opp = 21;
   stencil_[22].dir[0] = -1;  stencil_[22].dir[1] = 1;  stencil_[22].dir[2] = 1;
   stencil_[22].c = 1.0/216.0;

   stencil_[23].opp = 20;
   stencil_[23].dir[0] = -1;  stencil_[23].dir[1] = 1;  stencil_[23].dir[2] = -1;
   stencil_[23].c = 1.0/216.0;

   stencil_[24].opp = 19;
   stencil_[24].dir[0] = -1;  stencil_[24].dir[1] = -1;  stencil_[24].dir[2] = 1;
   stencil_[24].c = 1.0/216.0;

   stencil_[25].opp = 18;
   stencil_[25].dir[0] = -1;  stencil_[25].dir[1] = -1;  stencil_[25].dir[2] = -1;
   stencil_[25].c = 1.0/216.0;

   stencil_[26].opp = 26;
   stencil_[26].dir[0] = 0;  stencil_[26].dir[1] = 0;  stencil_[26].dir[2] = 0;
   stencil_[26].c = 8.0/27.0;

}
