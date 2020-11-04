/*
 * Project 2: Performance Optimization
 */

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#if !defined(_MSC_VER)
#include <pthread.h>
#endif 

#include <omp.h> 
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "calc_depth_naive.h"
#include "calc_depth_optimized.h"
#include "utils.h"

float displacement(int dx, int dy) {
    return sqrt(dx * dx + dy * dy);
}

float square_eucl_dist(float a, float b) {
    int diff = a - b;
    return diff * diff;
}

__m128 sse_eucl_dist(float *left, float *right , __m128 total) {
    __m128 diff = _mm_sub_ps(_mm_loadu_ps(left), _mm_loadu_ps(right));
    total = _mm_add_ps(_mm_mul_ps(diff, diff), total);
    return total;
}

void calc_depth_optimized(float *depth, float *left, float *right,
        int image_width, int image_height, int feature_width,
        int feature_height, int maximum_displacement) {

    int even_feature = feature_width % 2;

    /* goes through the image by going down each column as it iterates through the rows 
        (I think it’ll hit the cache more often this way). 
        Does the same thing for each box (displacement and feature)*/
    #pragma omp parallel for
    for (int x = 0; x < image_width; x++) {
        for (int y = 0; y < image_height; y++) {
            if (y < feature_height || y >= image_height - feature_height
                    || x < feature_width || x >= image_width - feature_width) {
                depth[y * image_width + x] = 0;
                continue;
            }
            float min_diff = -1;
            int min_dy = 0;
            int min_dx = 0;
            for (int dx = -maximum_displacement; dx <= maximum_displacement; dx++) {
                for (int dy = -maximum_displacement; dy <= maximum_displacement; dy++) {
                    if (y + dy - feature_height < 0
                            || y + dy + feature_height >= image_height
                            || x + dx - feature_width < 0
                            || x + dx + feature_width >= image_width) {
                        continue;
                    }
                    float vals[4] = {0, 0, 0, 0};
                    float squared_diff = 0;
                    __m128 total = _mm_setzero_ps();
                    /*for the feature it increments through the rows by 4 as it goes down the column. 
                        This is because of the sse vectorization it will take [i, i + 3] of the row its at. 
                        Performs squared euclidean distance and adds it to total. 
                        Which is then added to squared_diff.  */
                    for (int box_x = -feature_width; box_x <= feature_width - 4; box_x += 4) {
                        for (int box_y = -feature_height; box_y <= feature_height; box_y++) {
                            int left_x = x + box_x;
                            int left_y = y + box_y;
                            int right_x = x + dx + box_x;
                            int right_y = y + dy + box_y;
                            total = sse_eucl_dist(left + (left_y * image_width + left_x)
                                , right + (right_y * image_width + right_x), total);
                        }
                    }
                    _mm_storeu_ps(vals, total);
                    squared_diff += vals[0] + vals[1] + vals[2] + vals[3];
                    if (min_diff != -1) {
                        if (squared_diff > min_diff) {
                            continue;
                        }
                    }
                    /*If it is still less than the min displacement or min displacement hasn’t been set then it checks if the feature_width 
                        (num of rows) is even. If it isn’t then the vectorization earlier missed the last three values. 
                        Calculate it using same method but keeping the row addition to the left_x and right_x the same. 
                        Only add 1-3 of vals this time since vals[0] was already calculated in the last loop. 
                        If it’s even will just add last missing column values to 
                        add to squared_diff using the naive way of calculating squared_euclidean_distance. */
                    int tail_box_y = -feature_height;
                    int tail_box_x = 0;
                    if (even_feature) {
                        total = _mm_setzero_ps();
                        tail_box_x = feature_width - 3;
                    } else {
                        tail_box_x = feature_width;
                    }
                    while (tail_box_y <= feature_height) {
                            int left_x = x + tail_box_x;
                            int left_y = y + tail_box_y;
                            int right_x = x + dx + tail_box_x;
                            int right_y = y + dy + tail_box_y;
                            if (even_feature) {
                                total = sse_eucl_dist(left + (left_y * image_width + left_x), 
                                    right + (right_y * image_width + right_x), total); 
                            } else {
                                squared_diff += square_eucl_dist(left[left_y * image_width + left_x], 
                                    right[right_y * image_width + right_x]);
                            }
                            tail_box_y++;

                    }
                    if (even_feature) {
                        _mm_storeu_ps(vals, total);
                        squared_diff += vals[1] + vals[2] + vals[3];
                    }
                    if (min_diff == -1 || min_diff > squared_diff
                            || (min_diff == squared_diff
                                && displacement(dx, dy) < displacement(min_dx, min_dy))) {
                        min_diff = squared_diff;
                        min_dx = dx;
                        min_dy = dy;
                    }
                }
            }
            if (min_diff != -1) {
                if (maximum_displacement == 0) {
                    depth[y * image_width + x] = 0;
                } else {
                    depth[y * image_width + x] = displacement(min_dx, min_dy);
                }
            } else {
                depth[y * image_width + x] = 0;
            }
        }
    }
}
