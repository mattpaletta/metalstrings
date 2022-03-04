//
//  main.swift
//  runner
//
//  Created by Matthew Paletta on 2022-02-27.
//

import Foundation
/**
 * Create String struct, in a C-header
 * Import that into metal and Objective-C++
 * Copy list of strings to buffer (Size + static_cast<int>(chars))
 * Implement strcmp + substr in metal
 * use parallel prefix sum to insert into output buffer
 * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 * https://cs.wmich.edu/gupta/teaching/cs5260/5260Sp15web/lectureNotes/thm14%20-%20parallel%20prefix%20from%20Ottman.pdf
 * cast to std::string on CPU side
 */

PrefixSumTester().testPrefixSum()
