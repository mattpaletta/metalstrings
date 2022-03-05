//
//  metal_strings.h
//  string
//
//  Created by Matthew Paletta on 2022-03-04.
//

#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>

#define METAL_CONSTANT constant
#define METAL_DEVICE device
#define METAL_THREAD thread
#define METAL_THREADGROUP threadgroup
#define CPP_RESTRICT
#else
#define METAL_CONSTANT
#define METAL_DEVICE
#define METAL_THREAD
#define METAL_THREADGROUP
#define CPP_RESTRICT restrict
#endif

namespace metal {
#ifdef __METAL_VERSION__
    namespace memory {
        template<typename T>
        void memcpyThreadgroupDevice(device T* destination, const device T* source, size_t destIndex, size_t sourceIndex) {
            destination[destIndex] = source[sourceIndex];
            threadgroup_barrier(mem_flags::mem_device);
        }

        template<typename T>
        void memcpyThreadgroupDevice(device T* destination, const threadgroup T* source, size_t index) {
            memcpyThreadgroupDevice(destination, source, index, index);
        }

        template<typename T>
        void memcpyDeviceThreadgroup(threadgroup T* destination, const device T* source, size_t destIndex, size_t sourceIndex) {
            destination[destIndex] = source[sourceIndex];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        template<typename T>
        void memcpyDeviceThreadgroup(threadgroup T* destination, const device T* source, size_t index) {
            memcpyDeviceThreadgroup(destination, source, index, index);
        }

        template<typename T>
        void memcpyDeviceThread(thread T* destination, const device T* source, size_t sourceIndex) {
            *destination = source[sourceIndex];
            simdgroup_barrier(mem_flags::mem_none);
        }

        template<typename T>
        void memcpyThreadDevice(device T* destination, const thread T* source, size_t destIndex) {
            destination[destIndex] = *source;
            simdgroup_barrier(mem_flags::mem_none);
        }
    }
#endif

    namespace strings {
        METAL_CONSTANT char NULL_CHAR = '\0';

        int strcmp(const METAL_DEVICE char* str1, const METAL_DEVICE char* str2) {
            const METAL_DEVICE char* currChar1 = str1;
            const METAL_DEVICE char* currChar2 = str2;

            while (currChar1 && currChar2 && *currChar1 != NULL_CHAR && *currChar2 != NULL_CHAR) {
                if (*currChar1 > *currChar2) {
                    return 1;
                } else if (*currChar1 < *currChar2) {
                    return -1;
                }

                currChar1++;
                currChar2++;
            }

            if (currChar1 && currChar2 && *currChar1 == NULL_CHAR && *currChar2 == NULL_CHAR) {
                return 0;
            } else if (currChar1 && *currChar1 == NULL_CHAR) {
                return 1;
            } else {
                return -1;
            }
        }

        METAL_DEVICE char* strncpy(METAL_DEVICE char* /* restrict */ destination, const METAL_DEVICE char* /* restrict */ source, size_t num) {
            const METAL_DEVICE char* sourcePtr = source;
            METAL_DEVICE char* destinationCpy = destination;

            for (size_t i = 0; i < num; ++i) {
                *destinationCpy = *sourcePtr;
                destinationCpy++;
                sourcePtr++;
            }
            *destinationCpy = NULL_CHAR;

            return destination;
        }

        METAL_DEVICE char* strcpy(METAL_DEVICE char* /* restrict */ destination, const METAL_DEVICE char* /* restrict */ source) {
            const METAL_DEVICE char* sourcePtr = source;
            METAL_DEVICE char* destinationCpy = destination;

            while (sourcePtr && *sourcePtr != NULL_CHAR) {
                *destinationCpy = *sourcePtr;
                destinationCpy++;
                sourcePtr++;
            }
            *destinationCpy = NULL_CHAR;

            return destination;
        }

        METAL_DEVICE char* strncat(METAL_DEVICE char* /* restrict */ destination, const METAL_DEVICE char* /* restrict */ source, size_t num) {
            METAL_DEVICE char* endOfDest = destination;

            // Move to the end of destination
            while (endOfDest && *endOfDest++ != NULL_CHAR) {}

            // copy in the string
            return strncpy(endOfDest, source, num);
        }

        METAL_DEVICE char* strcat(METAL_DEVICE char* /* restrict */ destination, const METAL_DEVICE char* /* restrict */ source) {
            METAL_DEVICE char* endOfDest = destination;

            // Move to the end of destination
            while (endOfDest && *endOfDest++ != NULL_CHAR) {}

            // copy in the string
            return strcpy(endOfDest, source);
        }

        size_t strlen(const METAL_DEVICE char* str) {
            size_t size = 0;
            const METAL_DEVICE char* strPtr = str;
            while (strPtr && *strPtr != NULL_CHAR) {
                ++size;
            }

            return size;
        }

        const METAL_DEVICE char* strchr(const METAL_DEVICE char* str, int character) {
            const char charToFind = character;
            const METAL_DEVICE char* strPtr = str;
            while (strPtr && *strPtr != NULL_CHAR) {
                if (*strPtr == charToFind) {
                    return strPtr;
                }
            }

            return nullptr;
        }
    }
}
