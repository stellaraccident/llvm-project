include_directories(
  ${CMAKE_SOURCE_DIR}/lib/Target/X86
  ${CMAKE_BINARY_DIR}/lib/Target/X86
  )

set(LLVM_LINK_COMPONENTS
  Analysis
  CodeGen
  Core
  MC
  MIRParser
  Support
  Target
  TargetX86
  )

add_llvm_unittest(X86Tests
  MachineSizeOptsTest.cpp
)
