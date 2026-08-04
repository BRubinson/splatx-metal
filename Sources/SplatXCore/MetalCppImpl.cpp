// Emits the metal-cpp implementation — this TU is the only one in the whole project that defines
// *_PRIVATE_IMPLEMENTATION (metal-cpp README convention: a single definition point; duplicating it
// collides as link symbols).
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <Metal/Metal.hpp>
