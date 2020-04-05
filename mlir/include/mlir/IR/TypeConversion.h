//===- TypeConversion.h - Type conversion patterns ----==--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPE_CONVERSION_H
#define MLIR_IR_TYPE_CONVERSION_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

class OpBuilder;

/// Abstract class representing conversions from a 'source' type to 'target'
/// types. Hooks are also provided to insert 1:N casts from 'source' to
/// 'target' and N:1 casts from 'targets' to 'source'. In this way, a fully
/// specified TypeConversionPattern can allow for the materialization of local,
/// reversible casts in and out of a target type.
///
/// An individual instance can represent multiple source type conversions
/// so long as the source to target mapping is unambiguous for all hooks.
class TypeConversionPattern {
public:
  virtual ~TypeConversionPattern() = default;

  // Design note (do not submit):
  // Some things might be nicer if we had a ConversionResult struct which
  // encapsulates the conversion and is passed to the cast methods. I've found
  // that as factored, it is fine for immediate use, but we end up with snaking
  // of loose values in dialect conversion if we need to re-associate this
  // state later. Then convertType could just return an 
  // Optional<ConversionResult>. Then in dialect conversion, where we currently
  // just pass loose target types around, we could pass a ConversionResult
  // which would let us "get back" and do casts.
  struct ConversionResult {
    const TypeConversionPattern *pattern;
    Type sourceType;
    SmallVector<Type, 2> targetTypes;
  }

  /// Converts a source type to zero or more target types.
  /// Implementations should do one of the following:
  ///   * Return `llvm::None` if the pattern elects to not handle this type
  ///     conversion, in which case, the caller is free to try other patterns
  ///     it may have available.
  ///   * Return `success()` and populate `targetTypeResults` on a successful
  ///     conversion.
  ///   * Return `failure()` on an authoritative unsuccessful conversion.
  ///
  /// In the success case, `targetTypeResults` should be populated with:
  ///   * No target types: This signifies that any value associated with the
  ///     type should be removed, and it is up to other infrastructure to
  ///     ensure that this is valid.
  ///   * One target type that is the same as the source type: Such an identity
  ///     conversion will not result in calls to `castToSource` or
  ///     `castToTarget` and signifies that a type is unchanged.
  ///   * One or more target types (different from the source): Type conversion
  ///     should take place by calling `castToSource` or `castToTarget` as
  ///     appropriate.
  virtual Optional<LogicalResult>
  convertType(Type sourceType,
              SmallVectorImpl<Type> &targetTypeResults) const = 0;

  /// Inserts casts to convert a `sourceValue` into a list of target values
  /// where it is expected that there will be one target value produced for each
  /// type populated in a corresponding call to convertType (passed as
  /// `targetTypes`).
  /// Returns whether `targetValues` has been populated.
  /// The default implementation returns failure.
  virtual LogicalResult castToTarget(Location loc, Value sourceValue,
                                     ArrayRef<Type> targetTypes,
                                     SmallVectorImpl<Value> &targetValueResults,
                                     OpBuilder &builder) const {
    return failure();
  }

  /// Inserts casts to convert a list of `targetValues` (corresponding to a
  /// previous call to `convertType`) to the given `sourceType`.
  /// Returns a null value on a failure to cast.
  /// The default implementation returns a null value.
  virtual Value castToSource(Location loc, Type sourceType,
                             ArrayRef<Value> targetValues,
                             OpBuilder &builder) const {
    return nullptr;
  }
};

/// A list of TypeConversionPatterns.
class OwningTypeConversionPatternList {
  using PatternListT = SmallVector<std::unique_ptr<TypeConversionPattern>, 4>;

public:
  PatternListT::iterator begin() { return patterns.begin(); }
  PatternListT::iterator end() { return patterns.end(); }
  PatternListT::const_iterator begin() const { return patterns.begin(); }
  PatternListT::const_iterator end() const { return patterns.end(); }
  void clear() { patterns.clear(); }

  /// Add an instance of each of the pattern types 'Ts' to the pattern list with
  /// the given arguments.
  /// Note: ConstructorArg is necessary here to separate the two variadic lists.
  template <typename... Ts, typename ConstructorArg,
            typename... ConstructorArgs,
            typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void insert(ConstructorArg &&arg, ConstructorArgs &&... args) {
    // The following expands a call to emplace_back for each of the pattern
    // types 'Ts'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    using dummy = int[];
    (void)dummy{
        0, (patterns.emplace_back(std::make_unique<Ts>(arg, args...)), 0)...};
  }

  /// Insert an already-constructed pattern.
  void insertExisting(std::unique_ptr<TypeConversionPattern> pattern) {
    patterns.push_back(std::move(pattern));
  }

private:
  PatternListT patterns;
};

} // end namespace mlir

#endif // MLIR_IR_TYPE_CONVERSION_H
