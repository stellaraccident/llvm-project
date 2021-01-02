set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Object
  Support
  Symbolize
  )

add_llvm_tool(llvm-cfi-verify
  llvm-cfi-verify.cpp
  )

add_subdirectory(lib)
# TODO: Can be dynamic linked once LLVMCFIVerify is converted to a component
# library.
set_target_properties(llvm-cfi-verify PROPERTIES LLVM_LINK_STATIC ON)
target_link_libraries(llvm-cfi-verify PRIVATE LLVMCFIVerify)
