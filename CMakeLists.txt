CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# find deal.II
MESSAGE("Find deal.II")
FIND_PACKAGE(deal.II 9.7 REQUIRED
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
        )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT("fe point evaluation benchmarks")

# find likwid
MESSAGE("Find Likwid")
FIND_LIBRARY(LIKWID likwid HINTS ${LIKWID_DIR}/lib)
MESSAGE(STATUS "Using LIKWID found at ${LIKWID}")

FILE(GLOB FILES "performance_*.cc")
FOREACH(FILE ${FILES})
    GET_FILENAME_COMPONENT(APP ${FILE} NAME_WLE)

    ADD_EXECUTABLE(${APP} ${FILE})

    IF(EXISTS ${LIKWID})
        # The macro LIKWID_PERFMON is used inside Likwid, i.e., we need to pass
        # that precise macro into the Likwid machinery for compilation.
        TARGET_COMPILE_DEFINITIONS(${APP} PUBLIC LIKWID_PERFMON)
        TARGET_INCLUDE_DIRECTORIES(${APP} PUBLIC ${LIKWID_DIR}/include)
        TARGET_LINK_LIBRARIES(${APP} ${LIKWID})
    ENDIF()

    DEAL_II_SETUP_TARGET(${APP})
ENDFOREACH()
