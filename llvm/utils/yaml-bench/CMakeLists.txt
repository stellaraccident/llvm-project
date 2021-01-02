set(LLVM_LINK_COMPONENTS
  Support
)
add_llvm_utility(yaml-bench
  YAMLBench.cpp
  )

#target_link_libraries(yaml-bench PRIVATE LLVMSupport)
