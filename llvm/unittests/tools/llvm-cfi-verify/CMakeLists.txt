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

add_llvm_unittest(CFIVerifyTests
  FileAnalysis.cpp
  GraphBuilder.cpp
  )
# TODO: Can be dynamic linked once LLVMCFIVerify is converted to a component
# library.
set_target_properties(CFIVerifyTests PROPERTIES LLVM_LINK_STATIC ON)
target_link_libraries(CFIVerifyTests PRIVATE LLVMCFIVerify)
