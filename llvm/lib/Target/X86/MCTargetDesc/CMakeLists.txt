add_llvm_component_library(LLVMX86Desc
  X86ATTInstPrinter.cpp
  X86IntelInstPrinter.cpp
  X86InstComments.cpp
  X86InstPrinterCommon.cpp
  X86ShuffleDecode.cpp
  X86AsmBackend.cpp
  X86MCTargetDesc.cpp
  X86MCAsmInfo.cpp
  X86MCCodeEmitter.cpp
  X86MachObjectWriter.cpp
  X86ELFObjectWriter.cpp
  X86WinCOFFObjectWriter.cpp
  X86WinCOFFStreamer.cpp
  X86WinCOFFTargetStreamer.cpp

  LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  # TODO: Intra-component linkage needs to be via LINK_LIBS, not LINK_COMPONENTS
  # X86Info
  BinaryFormat

  LINK_LIBS PRIVATE
  LLVMX86Info

  ADD_TO_COMPONENT
  X86
  )
