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
#
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
# `llvm_component_add_library` and specifying an `ADD_TO_COMPONENT <foo>`
# parameter. As long as one such library is added to a given component name,
# then the `llvm_component::<foo>` alias will be available to link against it.
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
# be considered an implementation detail and is generally only useful for
# setting compiler definitions directly on the objects that comprise this
# part of the component. Note that it is illegal to depend on one library that
# makes up a component from another in the same component (and unnecessary,
# since their objects will participate in the same shared or static library
# link at the component level).
#
# Internal layout:
# ----------------
# Internally, a number of targets are created. In the below `${libname}` refers
# to the library `name` passed to `llvm_component_add_library` and
# `${componentname}` refers to the argument `ADD_TO_COMPONENT`.
#
#   * `${libname}_impl`: An object library containing the objects that should be
#     contributed to the component.
#   * `${componentname}`: INTERFACE library that
#     `llvm-component::${componentname}` aliases to.
#   * `${componentname}_props` : Custom property target for the component.
#   * `${componentname}_shared` (on disk as `${componentname}`) :
#     The shared library for the component (if not redirected to an aggregate
#     library).
#   * `${componentname}_static` (on disk as `${componentname}_static`) :
#     The static library for the component. This library does not participate
#     in shared library redirection and will be used by executables and
#     libraries configured for static linking.
#
# These should all be considered implementation details and not depended on.

# Adds a library that is part of a component.
function(llvm_component_add_library name)
  cmake_parse_arguments(ARG
    "EXPORT_EXPLICIT_SYBMOLS"
    "ADD_TO_COMPONENT"
    "ADDITIONAL_HEADERS;DEPENDS;LINK_LIBS"
    ${ARGN})
  # Validate arguments.
  if(ARG_ADD_TO_COMPONENT)
    set(component_name ${ARG_ADD_TO_COMPONENT})
  else()
    message(FATAL_ERROR "ADD_TO_COMPONENT is required for ${name}")
  endif()

  # The main component implementation (object) library.
  set(_impl_target ${name}_impl)
  llvm_process_sources(_all_src_files
    ${ARG_UNPARSED_ARGUMENTS} ${ARG_ADDITIONAL_HEADERS})
  add_library(${_impl_target} OBJECT EXCLUDE_FROM_ALL
    ${_all_src_files}
  )
  llvm_update_compile_flags(${_impl_target})
  set_target_properties(${_impl_target} PROPERTIES FOLDER "Object Libraries")
  # Set a marker to indicate that the library is a component. This is used
  # by some validity checks to enforce depending on it in the right way.
  set_target_properties(${_impl_target} PROPERTIES
    LLVM_LIBRARY_IS_NEWCOMPONENT_LIB TRUE)

  # Explicit depends.
  if(ARG_DEPENDS)
    # TODO: Object library deps are non transitive.
    add_dependencies(${_impl_target} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    # TODO: Object library deps are non transitive.
    target_link_libraries(${_impl_target} PRIVATE ${ARG_LINK_LIBS})
  endif()

  # Set export visibility at the library level.
  if(ARG_EXPORT_EXPLICIT_SYBMOLS)
    set_property(TARGET ${_impl_target} CXX_VISIBILITY_PRESET "hidden")
    set_property(TARGET ${_impl_target} C_VISIBILITY_PRESET "hidden")
  endif()

  # Get or create the component.
  set(component_props_target ${component_name}_props)
  set(component_interface_target ${component_name})
  if(NOT TARGET ${component_interface_target})
    # Create the component.
    llvm_component_create(${component_name})
  endif()

  # Resolve the static and shared library sub-targets from the component.
  get_target_property(component_shared_target
    ${component_props_target} LLVM_NEWCOMPONENT_SHARED_TARGET)
  get_target_property(component_static_target
    ${component_props_target} LLVM_NEWCOMPONENT_STATIC_TARGET)

  # Add the impl library to component static and link libraries as needed.
  if(component_shared_target)
    if(ARG_DEPENDS)
      add_dependencies(${component_shared_target} ${ARG_DEPENDS})
    endif()
    target_sources(${component_shared_target} PRIVATE $<TARGET_OBJECTS:${_impl_target}>)
    # NOTE: A generator such as $<TARGET_PROPERTY:${_impl_target},LINK_LIBRARIES>
    # can make this property "live" based on the object library, but formulating
    # it properly requires more features than old versions of cmake have (and
    # is tricky).
    target_link_libraries(${component_shared_target}
      PRIVATE ${ARG_LINK_LIBS})
  endif()
  if(component_static_target)
    if(ARG_DEPENDS)
      add_dependencies(${component_static_target} ${ARG_DEPENDS})
    endif()
    target_sources(${component_static_target} PRIVATE $<TARGET_OBJECTS:${_impl_target}>)
    target_link_libraries(${component_static_target}
      PRIVATE ${ARG_LINK_LIBS})
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
  # TODO: Revisit this once the migration is complete and see if we can
  # eliminate this case.
  if(NOT ${name} STREQUAL ${component_name})
    add_library(llvm-component::${name} ALIAS ${component_interface_target})
  endif()

  # The compile targets are responsible for compiling the sources, and the
  # link targets are what is ultimately performing the link. In degenerate
  # cases, there can be multiples of each (i.e. on windows where libraries
  # destined for a DLL are compiled differently from those that are not).
  set_property(TARGET ${component_props_target}
    APPEND PROPERTY LLVM_NEWCOMPONENT_COMPILE_TARGETS ${_impl_target}
  )

  # Extend the interface link libraries appropriately.
  # Static linking just links directly against the static library.
  target_link_libraries(${component_interface_target} INTERFACE
    $<$<BOOL:$<TARGET_PROPERTY:LLVM_LINK_STATIC>>:${component_static_target}>
  )

  # Shared linking will either link against component shared library, if
  # it exists, or the TODO
  set(actual_shared_target ${component_shared_target})
  if(NOT component_shared_target)
    set(actual_shared_target ${component_static_target})
  endif()
  target_link_libraries(${component_interface_target} INTERFACE
    $<$<NOT:$<BOOL:$<TARGET_PROPERTY:LLVM_LINK_STATIC>>>:${actual_shared_target}>
  )
endfunction()

function(llvm_component_create component_name)
  set(component_props_target ${component_name}_props)
  set(component_interface_target ${component_name})

  # Sanity check.
  if(TARGET ${component_props_target} OR
     TARGET ${component_interface_target})
    message(FATAL_ERROR "Attempted to create component ${component_name} multiple times")
  endif()

  # Create the interface library for the component.
  add_library(${component_interface_target} INTERFACE)

  # And alias it for easy use.
  add_library(llvm-component::${component_name} ALIAS ${component_interface_target})

  # Add a dummy target that is just for holding component properties.
  # This is because INTERFACE libraries cannot have custom properties, and
  # we prefer to not randomly pollute the global namespace.
  add_custom_target(${component_props_target})

  # Set the interface target on the property target.
  set_target_properties(${component_props_target} PROPERTIES
    LLVM_NEWCOMPONENT_INTERFACE_TARGET ${component_interface_target}
  )

  # Locate the component shared library target that corresponds to this
  # component name.
  llvm_component_ensure_libraries(${_component_name})

  # Exports and creates install targets for the component.
  llvm_component_install(${_component_name})
endfunction()

# Finds the shared library target that corresponds to the requested
# `component_name`.
function(llvm_component_ensure_libraries component_name)
  set(component_props_target ${component_name}_props)
  llvm_component_create_dummy_source(_dummy_file ${component_name})

  # Ensure the shared target.
  get_target_property(_shared_taget
    ${component_props_target} LLVM_NEWCOMPONENT_SHARED_TARGET)
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
    set_property(TARGET ${component_props_target} PROPERTY
      LLVM_NEWCOMPONENT_SHARED_TARGET ${_shared_target})
    set_property(TARGET ${component_props_target} APPEND PROPERTY
      LLVM_NEWCOMPONENT_LINK_TARGETS ${_shared_target})
  endif()

  if(NOT TARGET ${_shared_target})
    # Not yet created. Do so now.
    # TODO: This doesn't quite work yet for redirection. Need to not set the
    # OUTPUT_NAME in that case.
    add_library(${_shared_target} SHARED
      ${_dummy_file})
    set_property(TARGET ${_shared_target} PROPERTY OUTPUT_NAME ${component_name})
  endif()

  # Ensure the static target. Static targets are always per-component and do
  # not participate in redirection.
  get_target_property(_static_taget
    ${component_props_target} LLVM_NEWCOMPONENT_STATIC_TARGET)
  if(NOT _static_target)
    # Latch it.
    set(_static_target "${component_name}_static")
    set_property(TARGET ${component_props_target} PROPERTY
      LLVM_NEWCOMPONENT_STATIC_TARGET ${_static_target})
    set_property(TARGET ${component_props_target} APPEND PROPERTY
      LLVM_NEWCOMPONENT_LINK_TARGETS ${_shared_target})
  endif()

  if(NOT TARGET ${_static_target})
    add_library(${_static_target} STATIC
      ${_dummy_file})
  endif()
endfunction()

function(llvm_component_install component_name)
  set(component_props_target ${component_name}_props)
  get_target_property(component_interface_target ${component_props_target}
    LLVM_NEWCOMPONENT_INTERFACE_TARGET)
  get_target_property(component_static_target ${component_props_target}
    LLVM_NEWCOMPONENT_STATIC_TARGET)
  get_target_property(component_shared_target ${component_props_target}
    LLVM_NEWCOMPONENT_SHARED_TARGET)

  set(export_to_llvmexports EXPORT LLVMExports)
  set_property(GLOBAL PROPERTY LLVM_HAS_EXPORTS True)

  # Create the main install target for the interface library.
  install(TARGETS ${component_interface_target}
    ${export_to_llvmexports}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${component_name}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${component_name}
    RUNTIME DESTINATION bin COMPONENT ${component_name})

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-${component_name}
                              DEPENDS
                                ${component_interface_target}
                              COMPONENT ${component_name})
  endif()
  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${component_interface_target})

  # Generate install targets for each sub-library.
  foreach(t ${component_shared_target} ${component_static_target})
    install(TARGETS ${t}
      ${export_to_llvmexports}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${t}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${t}
      RUNTIME DESTINATION bin COMPONENT ${t})

    if (NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${t}
                                DEPENDS
                                  ${t}
                                COMPONENT ${t})
      add_dependencies(install-${component_name} install-${t})
      add_dependencies(install-${component_name}-stripped install-${t}-stripped)
    endif()
    set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${t})
  endforeach()
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
