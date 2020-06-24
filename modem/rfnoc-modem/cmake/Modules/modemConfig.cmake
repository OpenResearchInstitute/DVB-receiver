INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_MODEM modem)

FIND_PATH(
    MODEM_INCLUDE_DIRS
    NAMES modem/api.h
    HINTS $ENV{MODEM_DIR}/include
        ${PC_MODEM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    MODEM_LIBRARIES
    NAMES gnuradio-modem
    HINTS $ENV{MODEM_DIR}/lib
        ${PC_MODEM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MODEM DEFAULT_MSG MODEM_LIBRARIES MODEM_INCLUDE_DIRS)
MARK_AS_ADVANCED(MODEM_LIBRARIES MODEM_INCLUDE_DIRS)

