add_llvm_component_library(LLVMX86AsmParser
  X86AsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Support

  ADD_TO_COMPONENT
  LLVMTargetX86
  )
