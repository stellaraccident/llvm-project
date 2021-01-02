# LLVMComponents.cmake
#
# In the LLVM build system, a "component" is essentially a public library that
# can either be statically or dynamically linked. In the latter case, the
# consumer of a component will be transparently directed to the appropriate
# shared library (or DLL) that houses the component (which may, and likely
# is different from how the library is physically defined in the build
# system).
#
# Using components:
# =================

# Linking to components:
# ----------------------
# Consumers link to a component in one of two ways:
#   a) With a special `llvm-component::` namespace prefixed to the component
#      name in target_link_libraries (or LINK_LIBS if using an llvm_add_library
#      derived macro). Example: `llvm-component::LLVMSupport`.
#   b) For "classic" LLVM component aliases (such as "Support", "Core", etc),
#      the following mechanisms will expand component aliases and add them to
#      the target_link_libraries as above:
#      - By listing is in `LINK_COMPONENTS` of an llvm_add_library derived macro.
#      - By setting a variable `LLVM_LINK_COMPONENTS` in the scope prior to
#        calling a macro/function that creates an LLVM library or executable.
#
# Static vs dynamic linkage:
# --------------------------
# If the global `LLVM_BUILD_SHARED_COMPONENTS` option is enabled, then shared
# libraries (or DLLs) will be created for all components that support shared
# linkage on the platform, and by default, executables will link against these
# shared libraries.
#
# Individual executables can opt in to always link statically by setting
# a target property `LLVM_LINK_STATIC` to true. Example:
#   `set_target_properties(FileCheck PROPERTIES LLVM_LINK_STATIC ON)`
#
# Per-component shared-libraries vs aggregate:
# --------------------------------------------
# By default if `LLVM_BUILD_SHARED_COMPONENTS` is enabled, every component will
# be built into its own shared library. Various other groupings are possible.
# As an example, `LLVM_BUILD_LLVM_DYLIB` will redirect many of the core LLVM
# components into a libLLVM.so file instead of breaking them into individual
# libraries. This is implemented on top of the Component Redirection feature
# (see below).
#
# "Classic" LLVM components:
# --------------------------
# The wrappers `add_llvm_component_group` and `add_llvm_component_library`
# exist for components under `llvm/lib` defined with classic naming (i.e.
# "Core", "Support", etc). The naming is transparently mapped to underlying
# calls to `llvm_component_add_library` and dependencies on `llvm-component::`
# namespaces.
#
# Visibility:
# -----------
# When a library is added to a shared component, it is annotated as either
# "export all symbols" or "export explicit symbols". The former is the
# traditional default when building shared libraries on Unix-like systems and
# the latter is the default behavior when building Windows DLLs. We default
# all libraries to "export all symbols" mode and allow opt-in to explicit mode
# with the flag EXPORT_EXPLICIT_SYMBOLS.
#
# On Unix, in "export all symbols" mode when performing a shared build,
# libraries are built with their VISIBILITY_PRESET left as "default" (takes
# any cmake global configure default that may be defined). It will be set to
# "hidden" when in EXPORT_EXPLICIT_SYMBOLS mode. Note that if built with default
# visibility globally set to hidden, then shared library builds may not
# function. However, if this is being done, it is assumed that the user knows
# what they intend (there are cases where such things are useful, if not
# particularly well defined).
#
# On Windows, in EXPORT_ALL_SYMBOLS mode, special care is taken to generate
# appropriate def files to mimic the Unix-like EXPORT_ALL_SYMBOLS behavior to
# the extent possible. A number of caveats apply.
#
# By making this an explicit toggle, libraries can progressively opt-in to more
# granulary visibility controls in a portable way, which generally leads to
# smaller, better optimized shared libraries with less private-API leakage.
#
# Mapping to LLVM_LINK_COMPONENTS:
# --------------------------------
# LLVM_LINK_COMPONENTS is implemented on top of this feature and is retained
# as-is within the llvm/lib tree for compatibility. There is a global (TODO)
# that switches between building a mondo libLLVM.so or emitting fine grained
# shared components that map to top-level link components.
#
# Implementing components:
# ========================
#
# Internally, components are an abstraction disjoint from the libraries that
# make them up. Multiple libraries can contribute to a single, named component,
# although, in many common cases, there is a 1:1 correspondance between library
# and component. The namespace is laid out to accomodate this.
#
# Adding a library to a component is done by invoking
# `llvm_component_add_library` (which delegates most of its processing to
# `llvm_add_library`) and specifying an `ADD_TO_COMPONENT <foo>` parameter.
# As long as one such library is added to a given component name, then the
# `llvm_component::<foo>` alias will be available to link against it.
#
# Internally, a component consists of a number of targets, with a few available
# for external use:
#   * Compilation targets: Can be used to set compilation flags. Obtain via
#     `llvm_component_get_compile_targets` once a component has been
#     instantiated.
#   * Linkage targets: Can be used to set linkage flags. Obtain via
#     `llvm_component_get_link_targets`.
#   * Property target: Generic target that is safe to stash any properties on
#     related to the component. Get via `llvm_component_get_props_target`.
#
# The library name specified in the `llvm_component_add_library` call should
# be considered an implementation detail and is only useful in the case of
# assembling multi-library components that have intra-component dependencies.
# In these cases, libraries within the same component can depend on each other
# via `target_link_libraries`/`LINK_LIBS` using this library name. Within a
# component, libraries must not depend on the public `llvm-component::`
# target for their component (and doing so will be flagged as a dependency
# cycle).
#
# Internal layout:
# ----------------
# Internally, a number of targets are created. In the below `${libname}` refers
# to the library `name` passed to `llvm_component_add_library` and
# `${componentname}` refers to the argument `ADD_TO_COMPONENT`.
#
#   * `${libname}` (on disk as `${libname}_static`): The library added to a
#     component. Always created as a `STATIC OBJECT` library on Unix-like
#     systems. May have other ancillary libraries created for Windows.
#   * `${componentname}_interface`: INTERFACE library that
#     `llvm-component::${componentname}` aliases to.
#   * `${componentname}_props` : Custom property target for the component.
#   * `${componentname}_shared` (on disk as `${componentname}`) :
#     The shared library for the component (if not redirected to an aggregate
#     library).
#
# These should all be considered implementation details and not depended on.

# Adds a library that is part of a component.
function(llvm_component_add_library name)
  cmake_parse_arguments(ARG
    "EXPORT_EXPLICIT_SYBMOLS"
    "ADD_TO_COMPONENT"
    ""
    ${ARGN})

  # Validate arguments.
  if(ARG_ADD_TO_COMPONENT)
    set(_component_name ${ARG_ADD_TO_COMPONENT})
  else()
    message(FATAL_ERROR "ADD_TO_COMPONENT is required for ${name}")
  endif()

  # The main component static library and object archive.
  set(_static_target ${name})
  llvm_add_library(${_static_target} STATIC OBJECT
    # Since we want any dynamic library (i.e. libMyComponent.so) to be the
    # primary name, move the static library to a _static suffixed name.
    OUTPUT_NAME "${name}_static"
    #EXPORT_NAME "${name}"
    ${ARG_UNPARSED_ARGUMENTS})
  # Set a marker to indicate that the library is a component. This is used
  # by some validity checks to enforce depending on it in the right way.
  set_target_properties(${_static_library} PROPERTIES
    LLVM_LIBRARY_IS_NEWCOMPONENT TRUE)

  # Create the interface library for the component, if it has not already
  # been created.
  set(_interface_target ${_component_name}_interface)
  if(NOT TARGET ${_interface_target})
    add_library(${_interface_target} INTERFACE)
    llvm_component_export_target(${_interface_target})
    # And alias it for easy use.
    add_library(llvm-component::${_component_name} ALIAS ${_interface_target})
  endif()

  # Add a dummy target that is just for holding component properties.
  # This is because INTERFACE libraries cannot have custom properties, and
  # we prefer to not randomly pollute the global namespace.
  llvm_component_get_props_target(_props_target ${_component_name})
  if(NOT TARGET ${_props_target})
    add_custom_target(${_props_target})
  endif()

  # When building a multi-library component, we will end up with cases where
  # the library name (i.e. LLVMX86Desc) differs from the component name
  # (i.e. X86). However, it is quite wide-spread for consumers to still want
  # to link to these sub-component libraries as if they were components in
  # their own right. In order to support this, we simply alias the sub-component
  # library name into the llvm-component:: namespace, pointing back to the
  # main component. This has the nice side-effect that if *within* a component,
  # you depend on a sub-component in such a way, CMake will detect a cycle in
  # the dependency graph and tell you explicitly what is wrong (an this only
  # affects *implementations* of a component, which are fine to hold to a
  # higher bar than consumers).
  if(NOT ${name} STREQUAL ${_component_name})
    add_library(llvm-component::${name} ALIAS ${_interface_target})
  endif()

  # The compile targets are responsible for compiling the sources, and the
  # link targets are what is ultimately performing the link. In degenerate
  # cases, there can be multiples of each (i.e. on windows where libraries
  # destined for a DLL are compiled differently from those that are not).
  set_target_properties(${_props_target} PROPERTIES
    LLVM_NEWCOMPONENT_INTERFACE_TARGET ${_interface_target}
    LLVM_NEWCOMPONENT_COMPILE_TARGETS obj.${name}
    LLVM_NEWCOMPONENT_LINK_TARGETS ${name}
  )

  # Locate the component shared library target that corresponds to this
  # component name.
  llvm_component_get_shared_target(_shared_target ${_component_name})
  if(_shared_target)
    set_property(TARGET ${_props_target} APPEND PROPERTY
      LLVM_NEWCOMPONENT_LINK_TARGETS ${_shared_target}
    )

    # Add objects to the shared library target sources.
    target_sources(${_shared_target}
      PRIVATE $<TARGET_OBJECTS:obj.${_static_target}>)

    # TODO: Propagate link libraries and link components from the library.
    get_target_property(_link_libraries ${_static_target} LINK_LIBRARIES)
    if(_link_libraries)
      target_link_libraries(${_shared_target} PRIVATE ${_link_libraries})
    endif()
  endif()

  # Set compile/link settings (uses utility functions as callers will to
  # verify).
  llvm_component_get_compile_targets(_compile_targets ${_component_name})
  llvm_component_get_link_targets(_link_targets ${_component_name})
  if(ARG_EXPORT_EXPLICIT_SYBMOLS)
    set_property(TARGET ${_compile_targets} CXX_VISIBILITY_PRESET "hidden")
    set_property(TARGET ${_compile_targets} C_VISIBILITY_PRESET "hidden")
  endif()

  # Export the static library.
  llvm_component_export_target(${_static_target} STATIC)

  # Extend the link libraries of the interface library appropriately.
  target_link_libraries(${_interface_target} INTERFACE
    $<$<BOOL:$<TARGET_PROPERTY:LLVM_LINK_STATIC>>:${_static_target}>
  )
  if(_shared_target)
    # Shared libraries enabled, non-static uses shared.
    target_link_libraries(${_interface_target} INTERFACE
      $<$<NOT:$<BOOL:$<TARGET_PROPERTY:LLVM_LINK_STATIC>>>:${_shared_target}>
    )
  else()
    # Shared libraries disabled, non-static is still static.
    target_link_libraries(${_interface_target} INTERFACE
      $<$<NOT:$<BOOL:$<TARGET_PROPERTY:LLVM_LINK_STATIC>>>:${_static_target}>
    )
  endif()
endfunction()

function(llvm_component_export_target name)
  cmake_parse_arguments(ARG
    "STATIC"
    ""
    ""
    ${ARGN})

  set(export_to_llvmexports)

  # Note: Shared library components are always exported. Static components
  # are exported conditionally.
  if(NOT ARG_STATIC OR ${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
      NOT LLVM_DISTRIBUTION_COMPONENTS)
    set(export_to_llvmexports EXPORT LLVMExports)
    set_property(GLOBAL PROPERTY LLVM_HAS_EXPORTS True)
  endif()

  install(TARGETS ${name}
    ${export_to_llvmexports}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${name}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${name}
    RUNTIME DESTINATION bin COMPONENT ${name})

    if (NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                               DEPENDS ${name}
                               COMPONENT ${name})
    endif()

  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${name})
endfunction()

# Finds the shared library target that corresponds to the requested
# `component_name`, setting it in `out_target_name`. Resolves to
# ${component_name}-NOTFOUND if shared library building is disabled.
function(llvm_component_get_shared_target out_target_name component_name)
  if(NOT LLVM_BUILD_SHARED_COMPONENTS)
    set(${_out_target_name} ${component_name}-NOTFOUND PARENT_SCOPE)
    return()
  endif()

  get_property(_shared_target GLOBAL
    PROPERTY LLVM_NEWCOMPONENT_${component_name}_SHARED_TARGET)
  if(NOT _shared_target)
    # See if there is a redirection entry for the component (i.e. specifying
    # that the component goes into a dedicated shared library).
    get_property(_shared_target GLOBAL
      PROPERTY LLVM_NEWCOMPONENT_${component_name}_REDIRECT_SHARED_TARGET)

    # Neither. Go ahead and create the default association now.
    if(NOT _shared_target)
      set(_shared_target "${component_name}_shared")
    endif()

    # Memorialize the decision.
    set_property(GLOBAL PROPERTY
      LLVM_NEWCOMPONENT_${component_name}_SHARED_TARGET ${_shared_target})
  endif()

  if (NOT TARGET ${_shared_target})
    # Not yet created. Do so now.
    # TODO: This doesn't quite work yet for redirection. Need to not set the
    # OUTPUT_NAME in that case.
    llvm_component_create_dummy_source(_dummy_file ${component_name})
    llvm_add_library(${_shared_target}
      OUTPUT_NAME ${component_name}
      SHARED PARTIAL_SOURCES_INTENDED
      ${_dummy_file})
  endif()

  set(${out_target_name} ${_shared_target} PARENT_SCOPE)
endfunction()

# Gets a target in a component responsible for holding generic properites about
# the component. The components system does nothing with this target but
# ensures it is present and isolated from real library targets.
function(llvm_component_get_props_target out_target name)
  set(${out_target} ${name}_props PARENT_SCOPE)
endfunction()

# Gets the list of compilation targets that a component produces. Compilation
# affecting properties should be manipulated on these targets.
function(llvm_component_get_compile_targets out_targets name)
  get_target_property(_compile_targets ${name}_props LLVM_NEWCOMPONENT_COMPILE_TARGETS)
  if(NOT _compile_targets)
    message(SEND_ERROR "Could not get component compile targets for ${name} (is it a component library?)")
  endif()
  set(${out_targets} "${_compile_targets}" PARENT_SCOPE)
endfunction()

# Gets the list of linkable targets that a component produces. Link-affecting
# properties should be manipulated on these targets.
function(llvm_component_get_link_targets out_targets name)
  get_target_property(_link_targets ${name}_props LLVM_NEWCOMPONENT_LINK_TARGETS)
  if(NOT _link_targets)
    message(SEND_ERROR "Could not get component link targets for ${name} (is it a component library?)")
  endif()
  set(${out_targets} "${_link_targets}" PARENT_SCOPE)
endfunction()

# Short-cut to call 'target_include_dirs' for all compilation targets in a
# component.
function(llvm_component_include_dirs name)
  llvm_component_get_compile_targets(_targets ${name})
  foreach(_target in ${_targets})
    target_include_directories(${_target} ${ARGN})
  endforeach()
endfunction()

# Utility to create a dummy/empty source file.
# TODO: Just drop such a file somewhere in the source tree.
function(llvm_component_create_dummy_source out_name basename)
  set(_dummy_file ${CMAKE_CURRENT_BINARY_DIR}/${basename}__DummyComponent.cpp)
  file(WRITE ${_dummy_file} "// This file intentionally empty\n")
  set_property(SOURCE ${_dummy_file} APPEND_STRING PROPERTY COMPILE_FLAGS
      "-Wno-empty-translation-unit")
  set(${out_name} ${_dummy_file} PARENT_SCOPE)
endfunction()
