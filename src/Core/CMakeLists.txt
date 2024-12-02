set(LIB_NAME Core)

file(
		GLOB_RECURSE LIB_SRC
		LIST_DIRECTORIES false
		CONFIGURE_DEPENDS
		"${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.h*"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cuh*"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.cu*"
		"${CMAKE_CURRENT_SOURCE_DIR}/*.inl*"
)

if (WIN32)
	foreach(SRC IN ITEMS ${LIB_SRC})
		get_filename_component(SRC_PATH "${SRC}" PATH)
		file(RELATIVE_PATH SRC_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SRC_PATH}")
		string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
		source_group("${GROUP_PATH}" FILES "${SRC}")
	endforeach()
endif()

add_library(${LIB_NAME} STATIC ${LIB_SRC})

add_compile_definitions(_ENABLE_EXTENDED_ALIGNED_STORAGE)

if (WIN32)
	set_target_properties(${LIB_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
elseif(UNIX)
	if (CMAKE_BUILD_TYPE MATCHES Debug)
		set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Debug")
	else()
		set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/Release")
	endif()
endif()

if (WIN32)
	target_compile_options(${LIB_NAME} PRIVATE "/wd 4819")
endif()

set_target_properties(${LIB_NAME} PROPERTIES FOLDER "Engine")

target_include_directories(${LIB_NAME} PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/Core>)