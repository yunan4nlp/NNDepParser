
include_directories(
  model
)

add_executable(NNDepParser NNDepParser.cpp)
target_link_libraries(NNDepParser ${LIBS})
