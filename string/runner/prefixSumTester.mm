//
//  prefixSumTester.m
//  runner
//
//  Created by Matthew Paletta on 2022-02-27.
//

#import "prefixSumTester.h"
#include "metal_strings.h"

#import <MetalKit/MetalKit.h>

#include <iostream>
#include <vector>

namespace {
    template<typename T>
    std::vector<T> generateVector(std::size_t size) {
        std::vector<T> output(size);
        for (std::size_t i = 0; i < size; ++i) {
            if (i % 10) {
                if (i % 5 > i % 3) {
                    output.at(i) = i % 6;
                } else {
                    output.at(i) = i % 8;
                }
            } else {
                if (i % 2) {
                    output.at(i) = i % 3;
                } else {
                    output.at(i) = i % 7;
                }
            }
        }

        return output;
    }

    template<typename T>
    std::vector<T> exclusiveScan(const std::vector<T>& lst) {
        std::vector<T> output(lst.size(), 0);
        for (int i = 1; i < lst.size(); ++i) {
            output.at(i) = output.at(i - 1) + lst.at(i - 1);
        }
        return output;
    }

    template<typename T>
    std::vector<T> prefixSum(const std::vector<T>& lst) {
        return exclusiveScan(lst);
    }

    std::vector<int> parallelPrefixSum(const std::vector<int>& lst) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Load library with .metal files
        id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
        if (defaultLibrary == nil) {
            std::cerr << "Failed to find the default library." << std::endl;
            return {};
        }

        // Get a pointer to the function
        NSError* functionError = nil;
        auto constants = [[MTLFunctionConstantValues alloc] init];
        int localAlgorithm = 0;
        int globalAlgorithm = 0;
        [constants setConstantValue:&localAlgorithm type:MTLDataTypeInt atIndex:0];
        [constants setConstantValue:&globalAlgorithm type:MTLDataTypeInt atIndex:1];
        id<MTLFunction> prefixSumFunction = [defaultLibrary newFunctionWithName:@"prefix_exclusive_scan_uint32" constantValues:constants error:&functionError];
        if (prefixSumFunction == nil) {
            std::cerr << "Failed to find prefixSum function." << std::endl;
            return {};
        }

        // A pipeline runs a single function, optionally manipulating the input data before running the function, and the output data afterwards.
        // Creating the pipeline finishes compiling the shader for the GPU.
        NSError* error = nil;
        auto pipeline = [device newComputePipelineStateWithFunction:prefixSumFunction error:&error];

        // Schedules commands
        auto commandQueue = [device newCommandQueue];

        auto mBufferA = [device newBufferWithLength:lst.size() * sizeof(int) options:MTLResourceStorageModeShared];
        auto mBufferB = [device newBufferWithLength:lst.size() * sizeof(int) options:MTLResourceStorageModeShared];
        auto mBufferC = [device newBufferWithLength:lst.size() * sizeof(int) options:MTLResourceStorageModeShared];
        auto mBufferD = [device newBufferWithLength:lst.size() * sizeof(int) options:MTLResourceStorageModePrivate];

        // Buffer B is input
        for (std::size_t i = 0; i < lst.size(); ++i) {
            ((int*)mBufferB.contents)[i] = lst.at(i);
        }

        auto commandBuffer = [commandQueue commandBuffer];


        // Specify grid size (1D grid)
        MTLSize gridSize = MTLSizeMake(lst.size(), 1, 1);

        // Metal subdivides the group into smaller groups called threadgroups.
        NSUInteger threadGroupSize = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > lst.size()) {
            threadGroupSize = lst.size();
        }


        std::cout << "Executing with Grid Size (" << gridSize.width << "," << gridSize.height << "," << gridSize.depth << ")" << std::endl;
        std::cout << "Executing with ThreadGroup Size (" << threadGroupSize << ",1,1)" << std::endl;

        // Causes the GPU to create a grid of threads to execute on the GPU.
        // Set the list of commands to the encoder, which are stored in the buffer, and then send.
        {
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

            [computeEncoder setComputePipelineState:pipeline];

            // Offset of 0, means it will start reading data from the item 0 of the buffer
            // AtIndex specifies the argument it maps to in metal.
            [computeEncoder setBuffer:mBufferA offset:0 atIndex:0];
            [computeEncoder setBuffer:mBufferB offset:0 atIndex:1];
            [computeEncoder setBuffer:mBufferD offset:0 atIndex:2];

            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [computeEncoder endEncoding];
        }

        {
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

            // Call it twice
            [computeEncoder setComputePipelineState:pipeline];
            [computeEncoder setBuffer:mBufferC offset:0 atIndex:0];
            [computeEncoder setBuffer:mBufferA offset:0 atIndex:1];
            [computeEncoder setBuffer:mBufferD offset:0 atIndex:2];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            // Run the buffer
            [computeEncoder endEncoding];
        }
        [commandBuffer commit];

        [commandBuffer waitUntilCompleted];

        // read back into output buffer
        std::vector<int> output(lst.size());
        int* contents = (int*) mBufferC.contents;
        for (std::size_t i = 0; i < lst.size(); ++i) {
            output.at(i) = contents[i];
        }
        return output;
    }
}

@implementation PrefixSumTester

-(void)testStrcmp {
    {
        char str1[15];
        char str2[15];

        strcpy(str1, "abcdef");
        strcpy(str2, "ABCDEF");

        const auto cppresult = strcmp(str1, str2);
        const auto metalresult = metal::strings::strcmp(str1, str2);
        assert(cppresult > 0 == metalresult > 0 || cppresult < 0 && metalresult < 0 || cppresult == metalresult);
    }
    {
        char str1[40];
        strcpy(str1,"copy successful");

        char str2[40];
        metal::strings::strcpy(str2, "copy successful");
        assert(strcmp(str1, str2) == 0);
    }
    {
        char str1[15];
        char str2[15];
        strcpy(str1, "abcdef");
        metal::strings::strcpy(str2, "abcdef");

        assert(strcmp(str1, str2) == 0);
    }
    {
        char str1[15];
        char str2[15];

        // TODO: Write test that compares each byte.
        // TODO: Copies the first num characters of source to destination. If the end of the source C string (which is signaled by a null-character) is found before num characters have been copied, destination is padded with zeros until a total of num characters have been written to it.
        // No null-character is implicitly appended at the end of destination if source is longer than num. Thus, in this case, destination shall not be considered a null terminated C string (reading it as such would overflow).
        // destination and source shall not overlap (see memmove for a safer alternative when overlapping).

        strncpy(str1, "abcdef", 3);
        metal::strings::strncpy(str2, "abcdef", 3);

        assert(strcmp(str1, str2) == 0);
    }
}

-(void)testPrefixSum {
    [self testStrcmp];

    return;

    auto lst = generateVector<int>(100'000'000);

    std::vector<int> lst1;
    std::vector<int> lst2;

    if (true) {
//        std::vector<int> lst = {3, 1, 7, 0, 4, 1, 6, 3};
        std::cout << "CPU Start" << std::endl;
        auto sum = prefixSum(lst);
        std::cout << "CPU Done" << std::endl;
        lst1 = sum;
    }
    if (false) {
        auto sum = prefixSum(lst);
        for (const auto& elem : sum) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    if (true) {
//        std::vector<int> lst = {3, 1, 7, 0, 4, 1, 6, 3};
        std::cout << "GPU Start" << std::endl;
        auto sum = parallelPrefixSum(lst);
        std::cout << "GPU Done" << std::endl;
        lst2 = sum;
    }

    bool isDifferent = false;
    for (std::size_t i = 0; i < lst1.size(); ++i) {
        int a = lst.at(i);
        int b = lst.at(i);
        if (a != b) {
            isDifferent = true;
            break;
        }
    }

    std::cout << (isDifferent ? "Different" : "Same") << std::endl;
}

@end
