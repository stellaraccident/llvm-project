add_llvm_component_library(LLVMX86Disassembler
  X86Disassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  Support

  LINK_LIBS PRIVATE
  #LLVMX86Info

  ADD_TO_COMPONENT
  X86
  )
