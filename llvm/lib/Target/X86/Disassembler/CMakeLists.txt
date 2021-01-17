add_llvm_component_library(LLVMX86Disassembler
  X86Disassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  Support

  ADD_TO_COMPONENT
  LLVMTargetX86
  )
