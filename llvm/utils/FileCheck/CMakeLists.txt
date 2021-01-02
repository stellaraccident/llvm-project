set(LLVM_LINK_COMPONENTS
  FileCheck
  Support
)
add_llvm_utility(FileCheck
  FileCheck.cpp
  )

set_target_properties(FileCheck PROPERTIES LLVM_LINK_STATIC ON)
#target_link_libraries(FileCheck PRIVATE LLVMFileCheck LLVMSupport)
