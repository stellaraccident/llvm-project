add_llvm_component_library(LLVMX86AsmParser
  X86AsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Support
  # TODO: Intra-component linkage needs to be via LINK_LIBS, not LINK_COMPONENTS
  # X86Desc
  # X86Info

  LINK_LIBS PRIVATE
  #LLVMX86Desc
  #LLVMX86Info

  ADD_TO_COMPONENT
  X86
  )
