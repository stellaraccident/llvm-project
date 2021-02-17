//===- mlir-linalg-ods-yaml-gen.cpp - Linalg ODS generation from yaml  ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an ODS (and C++) generator from a YAML form
// derived from the mathematical expression of linalg named ops. Typically a
// math oriented DSL will be used to export the essential representation to
// this form, and maintaining the SOT at the math level (versus recreating it
// in MLIR) is deemed to have systemic value.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"

using namespace mlir;

using llvm::yaml::Input;
using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::Output;
using llvm::yaml::ScalarEnumerationTraits;
using llvm::yaml::ScalarTraits;

#define DEBUG_TYPE "linalg-ods-gen"

//===----------------------------------------------------------------------===//
// Mapping structs (correspond to data types in the YAML description).
// TODO: Since this is a schema/part of the contract, it should be moved to
// a real header.
//===----------------------------------------------------------------------===//

namespace {

struct LinalgYAMLContext {
  MLIRContext *mlirContext;
};

struct LinalgOpMetadata {
  std::string name;
  std::string cppOpName;
  Optional<std::string> doc;
};

struct SerializedAffineMap {
  AffineMapAttr affineMapAttr;

  AffineMap affineMap() { return affineMapAttr.getValue(); }
};

enum class LinalgTensorUsageDef {
  input,
  output,
  temporary,
};

struct LinalgTensorDef {
  std::string name;
  LinalgTensorUsageDef usage;
  SerializedAffineMap shape;
};

enum class LinalgIteratorTypeDef {
  parallel,
  reduction,
};

struct LinalgIndexingMapsConfig {
  Optional<SmallVector<SerializedAffineMap>> staticIndexingMaps;
};

struct ScalarExpression;

struct ScalarApply {
  std::string fnName;
  // NOTE: Must be pure heap allocated container (not SmallVector)
  // due to recursive data type.
  std::vector<ScalarExpression> operands;
};

struct ScalarExpression {
  Optional<std::string> scalarArg;
  Optional<ScalarApply> scalarApply;
};

struct ScalarAssign {
  std::string arg;
  ScalarExpression value;
};

struct LinalgStructuredOpConfig {
  SmallVector<LinalgTensorDef> args;
  LinalgIndexingMapsConfig indexingMaps;
  SmallVector<LinalgIteratorTypeDef> iteratorTypes;
  SmallVector<ScalarAssign> assignments;
};

struct LinalgOpConfig {
  Optional<LinalgOpMetadata> metadata;
  Optional<LinalgStructuredOpConfig> structuredOp;
};

} // namespace

//===----------------------------------------------------------------------===//
// Mapping traits.
//===----------------------------------------------------------------------===//

LLVM_YAML_IS_SEQUENCE_VECTOR(LinalgTensorDef);
LLVM_YAML_IS_SEQUENCE_VECTOR(SerializedAffineMap);
LLVM_YAML_IS_SEQUENCE_VECTOR(LinalgIteratorTypeDef);
LLVM_YAML_IS_SEQUENCE_VECTOR(ScalarAssign);
LLVM_YAML_IS_SEQUENCE_VECTOR(ScalarExpression);
LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(LinalgOpConfig);

namespace llvm {
namespace yaml {

template <>
struct ScalarEnumerationTraits<LinalgTensorUsageDef> {
  static void enumeration(IO &io, LinalgTensorUsageDef &value) {
    io.enumCase(value, "input", LinalgTensorUsageDef::input);
    io.enumCase(value, "output", LinalgTensorUsageDef::output);
    io.enumCase(value, "temporary", LinalgTensorUsageDef::temporary);
  }
};

template <>
struct ScalarEnumerationTraits<LinalgIteratorTypeDef> {
  static void enumeration(IO &io, LinalgIteratorTypeDef &value) {
    io.enumCase(value, "parallel", LinalgIteratorTypeDef::parallel);
    io.enumCase(value, "reduction", LinalgIteratorTypeDef::reduction);
  }
};

template <>
struct MappingTraits<LinalgOpConfig> {
  static void mapping(IO &io, LinalgOpConfig &info) {
    io.mapOptional("metadata", info.metadata);
    io.mapOptional("structured_op", info.structuredOp);
  }
};

template <>
struct MappingTraits<LinalgOpMetadata> {
  static void mapping(IO &io, LinalgOpMetadata &info) {
    io.mapRequired("name", info.name);
    io.mapRequired("cpp_op_name", info.cppOpName);
    io.mapOptional("doc", info.doc);
  }
};

template <>
struct MappingTraits<LinalgIndexingMapsConfig> {
  static void mapping(IO &io, LinalgIndexingMapsConfig &info) {
    io.mapOptional("static_indexing_maps", info.staticIndexingMaps);
  }
};

template <>
struct MappingTraits<ScalarApply> {
  static void mapping(IO &io, ScalarApply &info) {
    io.mapRequired("fn_name", info.fnName);
    io.mapRequired("operands", info.operands);
  }
};

template <>
struct MappingTraits<ScalarExpression> {
  static void mapping(IO &io, ScalarExpression &info) {
    io.mapOptional("scalar_arg", info.scalarArg);
    io.mapOptional("scalar_apply", info.scalarApply);
  }
};

template <>
struct MappingTraits<ScalarAssign> {
  static void mapping(IO &io, ScalarAssign &info) {
    io.mapRequired("arg", info.arg);
    io.mapRequired("value", info.value);
  }
};

template <>
struct MappingTraits<LinalgStructuredOpConfig> {
  static void mapping(IO &io, LinalgStructuredOpConfig &info) {
    io.mapRequired("args", info.args);
    io.mapRequired("indexing_maps", info.indexingMaps);
    io.mapRequired("iterator_types", info.iteratorTypes);
    io.mapRequired("assignments", info.assignments);
  }
};

template <>
struct MappingTraits<LinalgTensorDef> {
  static void mapping(IO &io, LinalgTensorDef &info) {
    io.mapRequired("name", info.name);
    io.mapRequired("usage", info.usage);
    io.mapRequired("shape", info.shape);
  }
};

template <>
struct ScalarTraits<SerializedAffineMap> {
  static void output(const SerializedAffineMap &value, void *rawYamlContext,
                     raw_ostream &out) {
    assert(value.affineMapAttr);
    value.affineMapAttr.print(out);
  }
  static StringRef input(StringRef scalar, void *rawYamlContext,
                         SerializedAffineMap &value) {
    assert(rawYamlContext);
    auto *yamlContext = static_cast<LinalgYAMLContext *>(rawYamlContext);
    if (auto attr = mlir::parseAttribute(scalar, yamlContext->mlirContext)
                        .dyn_cast_or_null<AffineMapAttr>()) {
      value.affineMapAttr = attr;
    } else {
      if (!value.affineMapAttr || !value.affineMapAttr.isa<AffineMapAttr>())
        return "could not parse as an affine map attribute";
    }
    return StringRef();
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

} // namespace yaml
} // namespace llvm

namespace {

//===----------------------------------------------------------------------===//
// Generation utilities
//===----------------------------------------------------------------------===//

class GenerationContext {
public:
  GenerationContext(MLIRContext *context, raw_ostream *odsOut,
                    raw_ostream *defnOut)
      : context(context), loc(UnknownLoc::get(context)), odsOut(odsOut),
        defnOut(defnOut) {}

  MLIRContext *getContext() { return context; }

  void setLoc(Location loc) { this->loc = loc; }
  Location getLoc() { return loc; }

  bool shouldGenerateOds() { return odsOut; }
  bool shouldGenerateDefns() { return defnOut; }

  raw_ostream &odss() {
    assert(odsOut && "ODS stream not defined");
    return *odsOut;
  }

  raw_ostream &defns() {
    assert(defnOut && "Definition stream not defined");
    return *defnOut;
  }

private:
  MLIRContext *context;
  Location loc;
  raw_ostream *odsOut;
  raw_ostream *defnOut;
};

} // namespace

static std::string generateCppExpression(SerializedAffineMap self,
                                         StringRef contextName) {
  std::string printedStr;
  llvm::raw_string_ostream printedSs(printedStr);
  self.affineMapAttr.print(printedSs);
  printedSs.flush();

  static const char exprFormat[] =
      R"FMT(mlir::parseAttribute("{0}", {1}).cast<AffineMapAttr>().getValue())FMT";
  return llvm::formatv(exprFormat, printedStr, contextName);
}

// A single line banner format. Parameters:
// {0}: Single line comment
static const char bannerFormat[] = R"FMT(
//===----------------------------------------------------------------------===//
// {0}
//===----------------------------------------------------------------------===//
)FMT";

//===----------------------------------------------------------------------===//
// Named generic op generation.
// These ops map at most a single contraction that complies with the limitations
// of a linalg.generic.
//===----------------------------------------------------------------------===//

// Template for Linalg named ops' ODS definitions. Parameters:
// {0}: ODS/C++ op name
// {1}: assembly op mnemonic
// {2}: op interface list
// {3}: documentation (summary + description)
// {4}: op attribute list
// {5}: the number of arguments for the op region
// {6}: builder methods taking standalone attribute parameters
// {7}: additional methods for attributes used by indexing maps
static const char structuredOpOdsHeaderFormat[] = R"FMT(
//===----------------------------------------------------------------------===//
// Op definition for {0}
//===----------------------------------------------------------------------===//

def {0} : LinalgStructuredBase_Op<"{1}", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  SingleBlockImplicitTerminator<"YieldOp">
  /*extraInterfaces=*/{2}]> {
    {3}
    let arguments = (ins
      Variadic<AnyShaped>:$inputs,
      Variadic<AnyShaped>:$outputs{4}
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    // TODO: Add builders back in.
    let printer = [{{ return ::printNamedStructuredOp(p, *this); }];
    let parser = [{{
      return ::parseNamedStructuredOp<{0}>(parser, result/*TODO:, captures*/);
    }];
    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(Block &block, ValueRange captures);
      static std::function<void(Block &, ValueRange)> getRegionBuilder() {{
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();
      {7}
    }];
}
)FMT";

// The iterator_types() method implementation. Parameters:
// {0}: Class name
// {1}: Comma interleaved iterator type names.
static const char structuredOpIteratorTypesFormat[] =
    R"FMT(
ArrayAttr {0}::iterator_types() {
  return Builder(getContext()).getStrArrayAttr(SmallVector<StringRef>{{ {1} });
}
)FMT";

// Implementations of getCanonicalizationPatterns, fold and getEffects.
// Parameters:
// {0}: Class name
const char structuredOpCanonicalizersAndFoldersFormat[] = R"FMT(
void {0}::getCanonicalizationPatterns(
    OwningRewritePatternList &results,
    MLIRContext *context) {{
  results.insert<EraseDeadLinalgOp>();
  results.insert<FoldTensorCastOp>();
}
LogicalResult {0}::fold(ArrayRef<Attribute>,
                        SmallVectorImpl<OpFoldResult> &) {{
  return foldMemRefCast(*this);
}
void {0}::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {{
  getGenericEffectsImpl(effects,
    getOperation()->getResults(), getInputBuffers(), getOutputBuffers());
}
)FMT";

static LogicalResult generateNamedGenericOpOds(LinalgOpConfig &opConfig,
                                               GenerationContext &genContext) {
  if (!genContext.shouldGenerateOds())
    return success();

  raw_ostream &os = genContext.odss();

  std::string interfaceNameList;
  std::string attrList;
  std::string attrMethods;
  std::string attrBuilder;

  std::string doc;
  if (opConfig.metadata->doc) {
    const char *docFmt = R"FMT(
      let summary = [{ {0} }];
      let description = [{
        {1}
      }];
    )FMT";
    StringRef summary, description;
    std::tie(summary, description) =
        StringRef(*opConfig.metadata->doc).trim().split('\n');
    doc = llvm::formatv(docFmt, summary.trim(), description.trim());
  }

  os << llvm::formatv(structuredOpOdsHeaderFormat, opConfig.metadata->cppOpName,
                      opConfig.metadata->name, interfaceNameList, doc, attrList,
                      opConfig.structuredOp->args.size(), attrBuilder,
                      attrMethods);

  return success();
}

static LogicalResult
generateNamedGenericOpDefns(LinalgOpConfig &opConfig,
                            GenerationContext &genContext) {
  if (!genContext.shouldGenerateDefns())
    return success();

  raw_ostream &os = genContext.defns();
  StringRef className = opConfig.metadata->cppOpName;

  // Implementation banner.
  std::string bannerComment = llvm::formatv("Implementation of {0}", className);
  os << llvm::formatv(bannerFormat, bannerComment);

  // Reference iterators.
  {
    std::string iteratorsStr;
    llvm::raw_string_ostream ss(iteratorsStr);
    llvm::interleaveComma(opConfig.structuredOp->iteratorTypes, ss,
                          [&](LinalgIteratorTypeDef it) {
                            switch (it) {
                            case LinalgIteratorTypeDef::parallel:
                              ss << "getParallelIteratorTypeName()";
                              break;
                            case LinalgIteratorTypeDef::reduction:
                              ss << "getReductionIteratorTypeName()";
                              break;
                            }
                          });
    ss.flush();
    os << llvm::formatv(structuredOpIteratorTypesFormat, className,
                        iteratorsStr);
  }

  // Static indexing maps.
  if (auto &staticMaps =
          opConfig.structuredOp->indexingMaps.staticIndexingMaps) {
    if (staticMaps->empty())
      return emitError(genContext.getLoc()) << "op has no indexing maps";
    AffineMap firstMap = staticMaps->front().affineMap();

    // Symbol bindings.
    {
      // For each symbol, generate a declaration for it, either with an
      // AffineSymbolExpr or an AffineConstantExpr (if the symbol derives from
      // an attribute).
      // TODO: Implement attribute constants.
      // TODO: Possibly lift into a top-level method.
      static const char structuredOpSymbolBindingsFormat[] = R"FMT(
static SmallVector<AffineExpr> getSymbolBindings({0} self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
{1}
  return exprs;
}
)FMT";

      unsigned symbolCount = firstMap.getNumSymbols();
      SmallVector<std::string> symbolBindings;
      for (unsigned i = 0; i < symbolCount; ++i) {
        // TODO: Switch and emit constants for attribute bound symbols.
        symbolBindings.push_back(llvm::formatv(
            "  exprs.push_back(getAffineSymbolExpr({0}, context));", i));
      }
      std::string symbolBindingsStr;
      llvm::raw_string_ostream symbolBindingsSs(symbolBindingsStr);
      llvm::interleave(symbolBindings, symbolBindingsSs, "\n");
      symbolBindingsSs.flush();

      os << llvm::formatv(structuredOpSymbolBindingsFormat, className,
                          symbolBindingsStr);
    }

    // Indexing maps.
    {
      // Parameters:
      // {0}: Class name
      // {1}: Comma-separated list of dimension variable names.
      // {2}: Statements
      static const char structuredOpIndexingMapsFormat[] = R"FMT(
ArrayAttr {0}::indexing_maps() {
  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  // TODO: Not sure if we need this since the affine expressions come in
  // normalized with the right dims already.
  AffineExpr {1};
  bindDims(context, {1});
{2}
  return Builder(context).getAffineMapArrayAttr(maps);
}
)FMT";

      unsigned dimCount = firstMap.getNumDims();

      // Generate a comma-separated list of dim identifiers to be passed to
      // bindDims, ensuring tht AffineExpr identifiers are bound in the right
      // order to the proper AffineDimExpr.
      // This results in vars in scope like: d0, d1, d2...
      SmallVector<unsigned> dimIndices;
      for (unsigned i = 0; i < dimCount; ++i)
        dimIndices.push_back(i);
      std::string dimIdentsStr;
      llvm::raw_string_ostream dimIdentsSs(dimIdentsStr);
      llvm::interleaveComma(dimIndices, dimIdentsSs,
                            [&](unsigned i) { dimIdentsSs << "d" << i; });
      dimIdentsSs.flush();

      // Statements to add and simplify each affine map.
      SmallVector<std::string> stmts;
      for (auto &indexingMap : *staticMaps) {
        // TODO: Assert that dim and symbol count match the first.
        stmts.push_back(
            llvm::formatv("  maps.push_back({0});",
                          generateCppExpression(indexingMap, "context")));
        stmts.push_back(llvm::formatv(
            "  maps.back() = "
            "simplifyAffineMap(maps.back().replaceDimsAndSymbols({{}, "
            "symbolBindings, {0}, 0));",
            dimCount));
      }

      std::string stmtsStr;
      llvm::raw_string_ostream stmtsSs(stmtsStr);
      llvm::interleave(stmts, stmtsSs, "\n");

      os << llvm::formatv(structuredOpIndexingMapsFormat, className,
                          dimIdentsStr, stmtsStr);
    }
  } else {
    // TODO: Error.
  }

  // getNumRegionArgs()
  {
    // Generates a getNumRegionArgs() method. Parameters:
    // {0}: Class name
    // {1}: Number of region args
    static const char structuredOpGetNumRegionArgsFormat[] = R"FMT(
unsigned {0}::getNumRegionArgs() {{ return {1}; }
)FMT";
    os << llvm::formatv(structuredOpGetNumRegionArgsFormat, className,
                        3 /* TODO DO NOT SUBMIT */);
  }

  // getLibraryCallName()
  {
    // Generates a getLibraryCallName method. Parameters:
    // {0}: Class name
    static const char structuredOpGetLibraryCallFormat[] = R"FMT(
std::string {0}::getLibraryCallName() {{
  return generateLibraryCallName(getOperation());
}
)FMT";
    os << llvm::formatv(structuredOpGetLibraryCallFormat, className);
  }

  // regionBuilder()
  {
    // Generates a regionBuilder method. Parameters.
    // {0}: Class name
    static const char structuredOpRegionBuilderFormat[] = R"FMT(
void {0}::regionBuilder(Block &block, ValueRange captures) {{
  // TODO
}
)FMT";
    os << llvm::formatv(structuredOpRegionBuilderFormat, className);
  }

  // Canonicalizers and folders.
  os << llvm::formatv(structuredOpCanonicalizersAndFoldersFormat, className);

  return success();
}

static LogicalResult generateOp(LinalgOpConfig &opConfig,
                                GenerationContext &genContext) {
  // Switch on op type being generated.
  if (opConfig.structuredOp) {
    return success(
        succeeded(generateNamedGenericOpOds(opConfig, genContext)) &&
        succeeded(generateNamedGenericOpDefns(opConfig, genContext)));
  } else {
    return emitError(genContext.getLoc()) << "unsupported operation type";
  }
}

//===----------------------------------------------------------------------===//
// Command line options and main
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory ODSGenCat("Linalg ODS Gen");

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("YAML filename"));

static llvm::cl::opt<std::string>
    outputOdsDeclFilename("o-ods-decl", llvm::cl::desc("ODS output filename"),
                          llvm::cl::value_desc("filename"), llvm::cl::init(""));

static llvm::cl::opt<std::string>
    outputCppImplFilename("o-impl",
                          llvm::cl::desc("C++ implementation file name"),
                          llvm::cl::value_desc("filename"), llvm::cl::init(""));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Linalg ODS Gen from YAML");

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  MLIRContext mlirContext;
  LinalgYAMLContext yamlContext{&mlirContext};

  std::vector<LinalgOpConfig> opConfigs;

  // Parse input.
  Input yin(file->getBuffer(), &yamlContext);
  yin >> opConfigs;

  if (yin.error())
    return 1;

  // Open output files.
  std::unique_ptr<llvm::ToolOutputFile> outputOdsDecl;
  if (!outputOdsDeclFilename.empty()) {
    outputOdsDecl = openOutputFile(outputOdsDeclFilename, &errorMessage);
    if (!outputOdsDecl) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  }

  std::unique_ptr<llvm::ToolOutputFile> outputCppImpl;
  if (!outputCppImplFilename.empty()) {
    outputCppImpl = openOutputFile(outputCppImplFilename, &errorMessage);
    if (!outputCppImpl) {
      llvm::errs() << errorMessage << "\n";
      return 1;
    }
  }

  if (!outputOdsDecl && !outputCppImpl) {
    llvm::errs() << "error: No output files specified\n";
    return 1;
  }

  // Generate.
  GenerationContext genContext(&mlirContext,
                               outputOdsDecl ? &outputOdsDecl->os() : nullptr,
                               outputCppImpl ? &outputCppImpl->os() : nullptr);

  for (auto &opConfig : opConfigs) {
    if (!opConfig.metadata) {
      emitError(genContext.getLoc())
          << "missing operation metadata on subsequent op";
      return 1;
    }

    genContext.setLoc(NameLoc::get(
        Identifier::get(opConfig.metadata->cppOpName, &mlirContext),
        &mlirContext));
    if (failed(generateOp(opConfig, genContext))) {
      return 1;
    }
  }

  if (outputOdsDecl)
    outputOdsDecl->keep();
  if (outputCppImpl)
    outputCppImpl->keep();

  return 0;
}
