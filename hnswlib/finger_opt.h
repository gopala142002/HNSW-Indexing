#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace hnswlib {

struct FingerNodeData {
    std::vector<float> neighbor_coeffs;
    std::vector<float> residual_norms;
    std::vector<float> neighbor_center_coeffs;
};

struct FingerGlobalData {
    int dim = 0;
    int rank = 0;
    bool ready = false;
    std::vector<float> projection;
    std::vector<float> point_projection;
    std::vector<float> point_norm_sq;
    float mu_true = 0.0f;
    float sigma_true = 1.0f;
    float mu_approx = 0.0f;
    float sigma_approx = 1.0f;
    float epsilon = 0.0f;
};

inline float finger_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

inline float finger_norm_sq(const float* x, int n) {
    return finger_dot(x, x, n);
}

inline float finger_cosine(const float* a, const float* b, int n) {
    float aa = finger_norm_sq(a, n);
    float bb = finger_norm_sq(b, n);
    if (aa <= 1e-20f || bb <= 1e-20f) return 0.0f;
    float v = finger_dot(a, b, n) / std::sqrt(aa * bb);
    return std::max(-1.0f, std::min(1.0f, v));
}

inline void finger_project(const float* x, const float* P, int dim, int rank, float* out) {
    for (int r = 0; r < rank; ++r) {
        float s = 0.0f;
        const float* row = P + static_cast<size_t>(r) * dim;
        for (int d = 0; d < dim; ++d) s += row[d] * x[d];
        out[r] = s;
    }
}

inline float finger_match_angle(float approx_angle, const FingerGlobalData& f) {
    float sigma = std::fabs(f.sigma_approx) > 1e-12f ? f.sigma_approx : 1.0f;
    float t = (approx_angle - f.mu_approx) * f.sigma_true / sigma + f.mu_true;
    t += f.epsilon;
    return std::max(-1.0f, std::min(1.0f, t));
}

inline float finger_approx_distance(
        float query_norm_sq,
        float center_norm_sq,
        float query_center_dist_sq,
        const float* query_projection,
        const float* center_projection,
        const float* neighbor_projection,
        float neighbor_residual_norm_sq,
        float neighbor_center_coeff,
        const FingerGlobalData& f) {

    float alpha = 0.0f;
    if (center_norm_sq > 1e-20f) {
        alpha = (query_norm_sq + center_norm_sq - query_center_dist_sq) /
                (2.0f * center_norm_sq);
    }

    float query_residual_norm_sq =
        std::max(0.0f, query_norm_sq - alpha * alpha * center_norm_sq);

    std::vector<float> query_residual_projection(f.rank);
    for (int r = 0; r < f.rank; ++r) {
        query_residual_projection[r] =
            query_projection[r] - alpha * center_projection[r];
    }

    float dot_projected = finger_dot(
        query_residual_projection.data(),
        neighbor_projection,
        f.rank);

    float qres_norm = std::sqrt(query_residual_norm_sq);
    float dres_norm = std::sqrt(std::max(0.0f, neighbor_residual_norm_sq));

    float approx_angle = 0.0f;
    if (qres_norm > 1e-20f && dres_norm > 1e-20f) {
        approx_angle = dot_projected / (qres_norm * dres_norm);
        approx_angle = std::max(-1.0f, std::min(1.0f, approx_angle));
    }

    float angle = finger_match_angle(approx_angle, f);

    float beta = neighbor_center_coeff;
    float projection_diff = alpha - beta;
    float projection_dist_sq = projection_diff * projection_diff * center_norm_sq;

    float result = projection_dist_sq + query_residual_norm_sq +
                   neighbor_residual_norm_sq -
                   2.0f * qres_norm * dres_norm * angle;

    return std::max(0.0f, result);
}

}