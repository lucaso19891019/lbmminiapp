

DFLAGS = -DSOA
INCLUDE = -I$(KOKKOS_HOME)/include -I../src -I../src/geometries -I../src/kernels -I../src/lattices -I../src/propagation
USERFLAGS = -cxx=icpx -O2 -fiopenmp
USERFLAGS += -fPIC -Wno-openmp-mapping -D__STRICT_ANSI__ -std=c++17 -fopenmp-targets=spir64#_gen -Xopenmp-target-backend "-device xehp"
USERFLAGS += $(DFLAGS) -DUSE_KOKKOS_OPENMPTARGET
CXXFLAGS = $(USERFLAGS) $(INCLUDE)
LINKFLAGS = $(USERFLAGS)
LIB=-L${KOKKOS_HOME}/lib64 -lkokkoscontainers -lkokkoscore

CXX = mpicxx
LINK = $(CXX)

OBJS =	Main.o \
	Communication.o \
	Parameters.o \
	Geometry.o GeometryAAAB.o \
	LatticeD3Q19.o LatticeD3Q27.o \
	Propagation.o PropagationAA.o PropagationAB.o \
	KernelAA_SOA_Base.o KernelAA_AOS_Base.o \
    KernelAA_SOA_Unroll.o  \
    KernelAB_SOA_Base.o KernelAB_AOS_Base.o \
    KernelAB_SOA_Pull.o \
    KernelAB_SOA_Unroll.o KernelAB_SOA_Unroll_Pull.o


builddir=test_build
FOBJS = $(addprefix $(builddir)/,$(OBJS))

lbm-proxy-app: $(FOBJS) 
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(LINK) $(LINKFLAGS) $(OBJS)  -o $@ $(LIB)

clean:
	/bin/mkdir -p $(builddir)
	cd $(builddir); rm -f *.o *.cuda *.host *.dat

# Compilation rules
$(builddir)/%.o:src/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<
$(builddir)/%.o:src/geometries/%.cpp 
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<
$(builddir)/%.o:src/kernels/%.cpp 
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<
$(builddir)/%.o:src/lattices/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<
$(builddir)/%.o:src/propagation/%.cpp
	/bin/mkdir -p $(builddir)
	cd $(builddir); $(CXX) $(CXXFLAGS) -c ../$<
