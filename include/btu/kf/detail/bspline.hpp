// Copyright (C) 2026 Jyggalag
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License, version 2 only,
// as published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// SPDX-License-Identifier: GPL-2.0-only
// See LICENSE.txt for the full license text.
//
// bspline.hpp - fit an open, uniform, clamped cubic (degree 3) B-spline to a
// sampled scalar/vector curve via least squares, matching the curve family
// used by NiBSplineData / NiBSplineBasisData in the Gamebryo/NetImmerse engine.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace bspline {

constexpr int kDegree = 3; // cubic, matches engine's B-spline interpolators

// Clamped uniform knot vector for M control points, degree p.
// Size = M + p + 1. First/last p+1 knots are 0/1 (clamped / "open").
inline std::vector<double> clampedUniformKnots(int M, int p) {
    int numKnots = M + p + 1;
    std::vector<double> k(numKnots);
    int interior = M - p - 1; // number of interior (non-clamped) knot values
    for (int i = 0; i <= p; ++i) k[i] = 0.0;
    for (int i = 0; i < interior; ++i)
        k[p + 1 + i] = double(i + 1) / double(interior + 1);
    for (int i = 0; i <= p; ++i) k[numKnots - 1 - i] = 1.0;
    return k;
}

// One sample's row of the basis matrix, stored sparsely. A degree-p basis
// row has at most p+1 nonzeros, all consecutive: N_{base..base+p}(t). The
// fit used to keep the full dense S x M matrix, which is O(S*M) memory and
// time -- harmless while M was capped at a few hundred, but with no cap a
// long clip (M and S both grow with duration) makes it quadratic: a
// 10-minute cutscene would need gigabytes per channel per worker thread.
// Every consumer below only ever reads these p+1 entries (the rest are
// exact zeros, and adding a zero never changes a floating-point sum), so
// the results are bit-identical to the dense version.
struct BasisRow {
    int base = 0;                 // index of v[0]; base..base+p are all < M
    double v[kDegree + 1] = {};
};

// Cox-de Boor recursive basis function N_{i,p}(t) via the standard
// dynamic-programming table (O(p^2) per evaluation).
inline BasisRow basisRow(const std::vector<double>& knots, int M, int p, double t) {
    int n = M - 1; // highest control point index
    if (t >= knots[n + 1]) t = std::nextafter(knots[n + 1], -1.0);
    if (t < knots[p]) t = knots[p];

    // Find the knot span: the i in [p, n] with knots[i] <= t < knots[i+1],
    // the last span being closed on the right. knots[p..n+1] is strictly
    // increasing for a clamped uniform vector, so a binary search finds the
    // same span the old linear scan did (that scan alone made every row
    // O(M)); a NaN t falls through to n in both.
    auto it = std::upper_bound(knots.begin() + p + 1, knots.begin() + n + 1, t);
    int span = int(it - knots.begin()) - 1;

    // Table of basis values via triangular recurrence.
    double N[kDegree + 1] = {};
    double left[kDegree + 1] = {}, right[kDegree + 1] = {};
    N[0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double denom = right[r + 1] + left[j - r];
            double temp = (denom != 0.0) ? N[r] / denom : 0.0;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
    BasisRow row;
    row.base = span - p;
    for (int j = 0; j <= p; ++j) row.v[j] = N[j];
    return row;
}

// Solve (A^T A) x = A^T b for several right-hand-side columns at once.
// `rhs` is stored column-major: rhs[col][row].
//
// The normal matrix is banded, so it is stored and factored as a band
// instead of a dense MxM matrix. Why it is banded: a degree-p B-spline
// basis row has at most p+1 consecutive nonzeros, so (A^T A)[i][j] == 0
// whenever |i-j| > p; the second-difference roughness penalty below only
// touches |i-j| <= 2. Half-bandwidth is therefore max(p, 2) == 3 for
// cubics. It is also symmetric positive definite (A^T A is PSD, and the
// 1e-9 diagonal plus the PSD penalty D^T D push it strictly positive), so
// a banded Cholesky factorization needs no pivoting and costs O(M * W^2)
// instead of the O(M^3) a dense Gauss-Jordan solve costs. That matters
// for --tolerance (adaptive M search) and large-M fits, where the solve
// used to dominate the runtime.
//
// Storage: band[i][k] holds ATA[i][i+k] (upper triangle, k = 0..W).
inline std::vector<std::vector<double>> solveNormalEquations(
    const std::vector<BasisRow>& A,                        // S rows, M columns
    size_t M,
    const std::vector<std::vector<double>>& rhsColumns) {  // C columns, each length S
    size_t S = A.size();
    size_t C = rhsColumns.size();
    const size_t W = size_t(std::max(kDegree, 2));

    // Build banded normal matrix ATA (M x (W+1)) and ATb (M x C). Each
    // basis row's nonzeros all sit inside [first, first+W], so only that
    // window is visited. `first` is the first NONZERO entry (an exact-knot
    // t can zero the leading value), matching the dense scan this replaced
    // so the accumulation order, and therefore every rounding, is unchanged.
    std::vector<std::vector<double>> band(M, std::vector<double>(W + 1, 0.0));
    std::vector<std::vector<double>> ATb(M, std::vector<double>(C, 0.0));
    for (size_t s = 0; s < S; ++s) {
        const BasisRow& row = A[s];
        const size_t base = size_t(row.base);
        auto at = [&](size_t i) { return i - base <= size_t(kDegree) ? row.v[i - base] : 0.0; };
        size_t first = base;
        while (first <= base + kDegree && row.v[first - base] == 0.0) ++first;
        if (first > base + kDegree) continue; // all-zero row contributes nothing
        size_t last = std::min(M, first + W + 1);
        for (size_t i = first; i < last; ++i) {
            double ai = at(i);
            if (ai == 0.0) continue;
            for (size_t j = i; j < last; ++j) band[i][j - i] += ai * at(j);
            for (size_t c = 0; c < C; ++c) ATb[i][c] += ai * rhsColumns[c][s];
        }
    }
    // Tiny Tikhonov regularization for numerical stability on
    // under-determined / degenerate systems (e.g. constant channels).
    for (size_t i = 0; i < M; ++i) band[i][0] += 1e-9;

    // Roughness penalty on the control polygon's discrete second
    // derivative (D^T D, D = second-difference operator). Required, not
    // cosmetic: the real engine's NiBSplineBasis::Compute hard-codes a
    // *uniform* clamped knot vector at decode time, so the encoder
    // cannot switch to chord-length
    // or otherwise non-uniform knot placement to fix this without
    // breaking playback -- this has to be solved on the solve side.
    //
    // Without it, whenever M ends up comparable to (or larger than) the
    // real sample count S -- which naturally happens on short/sparse
    // clips (sprint loops, jump-lands) once --min-control-points or the
    // ratio math pushes M up there -- the least-squares system stops
    // being safely overdetermined. Fitting a uniform-knot curve almost
    // exactly through samples it wasn't designed to space evenly against
    // is a classic Runge's-phenomenon setup: solved control points can
    // swing by 10s-100s of units between two unremarkable real values,
    // invisible at the sample points themselves (a near-exact fit is
    // near-perfect there by construction) but severe in between them at
    // actual playback. The existing 1e-9 term above does nothing to stop
    // this -- it's sized for avoiding exact singularity, not for damping
    // an ill-conditioned-but-technically-solvable system whose ATA
    // entries are real-world-position-scale (10s-1000s).
    //
    // Confirmed against real data pulled from a broken in-game sprint
    // animation: "Bip01 Rotate"'s translation channel with M equal to
    // its own sample count decoded to a ~2.6-unit swing between two
    // samples both near 0; a bone forced from its natural M up to the
    // --min-control-points floor (M=48 against 33 real samples) decoded
    // to a ~100-unit swing. Both dropped to well under 1 unit with this
    // penalty at the k below, with at-the-real-keyframe error staying in
    // the 0.001-0.03 unit range (negligible against curves spanning
    // 10s-100s of units) -- tuned empirically against both cases plus a
    // naturally-sized (M == S) fit, not guessed.
    //
    // Scaled relative to ATA's own mean diagonal (not a fixed absolute
    // value) so it stays negligible for safely-overdetermined fits
    // (M << S) and automatically tracks the channel's real unit scale
    // (position vs. quaternion component vs. scale) instead of behaving
    // differently per channel type the way a fixed constant would.
    if (M >= 3) {
        double diagSum = 0.0;
        for (size_t i = 0; i < M; ++i) diagSum += band[i][0];
        double lambda = (diagSum / double(M)) * 1e-3;
        if (lambda > 0.0) {
            // Upper-triangle half of each D^T D 3x3 block; the band is
            // symmetric so the lower entries are implied.
            for (size_t i = 0; i + 2 < M; ++i) {
                band[i][0]   += lambda;
                band[i][1]   -= 2.0 * lambda;
                band[i][2]   += lambda;
                band[i+1][0] += 4.0 * lambda;
                band[i+1][1] -= 2.0 * lambda;
                band[i+2][0] += lambda;
            }
        }
    }

    // Banded Cholesky: L[i][k] = L(i, i-k), the lower factor of
    // ATA = L L^T, k = 0..W. ATA(i, j) for j < i is read from the upper
    // storage as band[j][i-j].
    std::vector<std::vector<double>> L(M, std::vector<double>(W + 1, 0.0));
    for (size_t i = 0; i < M; ++i) {
        size_t jStart = i > W ? i - W : 0;
        for (size_t j = jStart; j < i; ++j) {
            double sum = band[j][i - j];
            for (size_t m = jStart; m < j; ++m) sum -= L[i][i - m] * L[j][j - m];
            L[i][i - j] = sum / L[j][0];
        }
        double d = band[i][0];
        for (size_t m = jStart; m < i; ++m) d -= L[i][i - m] * L[i][i - m];
        // SPD in exact arithmetic; guard anyway so a pathological rounding
        // case degrades like a tiny-pivot clamp instead of producing NaN.
        if (!(d > 1e-15)) d = 1e-15;
        L[i][0] = std::sqrt(d);
    }

    // Forward (L y = b) then back (L^T x = y) substitution per RHS column.
    std::vector<std::vector<double>> result(C, std::vector<double>(M));
    for (size_t c = 0; c < C; ++c) {
        auto& x = result[c];
        for (size_t i = 0; i < M; ++i) {
            double sum = ATb[i][c];
            size_t jStart = i > W ? i - W : 0;
            for (size_t j = jStart; j < i; ++j) sum -= L[i][i - j] * x[j];
            x[i] = sum / L[i][0];
        }
        for (size_t i = M; i-- > 0;) {
            double sum = x[i];
            size_t jEnd = std::min(M, i + W + 1);
            for (size_t j = i + 1; j < jEnd; ++j) sum -= L[j][j - i] * x[j];
            x[i] = sum / L[i][0];
        }
    }
    return result;
}

// Fit control points for `numComponents` interleaved channels (e.g. 3 for
// xyz, 4 for quaternion, 1 for scalar) given samples at parametric t in
// [0,1] with M control points. Returns control points as M rows of
// numComponents doubles.
struct FitResult {
    std::vector<std::vector<double>> controlPoints; // [M][numComponents]
    double maxError = 0.0;
};

inline FitResult fitBSpline(const std::vector<double>& ts,
                             const std::vector<std::vector<double>>& values, // [S][numComponents]
                             int M) {
    size_t S = ts.size();
    int p = kDegree;
    if (M < p + 1) M = p + 1;
    // Deliberately NOT clamping M down to S (sample count) here, even though
    // M > S makes the least-squares system underdetermined -- the Tikhonov
    // regularization below (ATA[i][i] += 1e-9) already makes that safely
    // solvable, and every caller of this function needs an EXACT M back,
    // not "whatever fewer points fit the data": when several channels of
    // one bone share a single basis block (see main.cpp's per-bone M
    // grouping), a sparse channel (e.g. a 2-key scale track) still has to
    // produce the SAME M as its bone's denser channels, or the control-point
    // pool ends up holding fewer elements for that channel's handle than
    // the shared basis block declares -- an out-of-bounds read at playback.

    auto knots = clampedUniformKnots(M, p);

    std::vector<BasisRow> A(S);
    for (size_t s = 0; s < S; ++s) A[s] = basisRow(knots, M, p, ts[s]);

    size_t C = values.empty() ? 0 : values[0].size();

    // Fit DEVIATIONS from each component's own mean, not the raw values.
    // Why: when M is large relative to S (a sparse channel -- e.g. a
    // 1-key scale track -- sharing a bone's basis block with a much denser
    // channel, forced up to that bone's shared M), most control points have
    // no real constraint from any sample and are set purely by the Tikhonov
    // regularization term (ATA[i][i] += 1e-9 below), which pulls an
    // unconstrained control point toward *zero*. For a scale channel
    // (nominally ~1.0) that means most of the curve decodes near 0 instead
    // of near 1, i.e. the model's scale collapses to nothing for most of
    // the clip. The same failure mode applies to rotation (droops to a
    // degenerate, non-unit quaternion instead of identity) and translation
    // (droops to world origin instead of the bone's own position).
    // Centering the fit on each component's mean makes the regularizer's
    // implicit default "hold this channel's own typical value" instead of
    // "hold zero" -- for the degenerate single-key case this is exact (mean
    // == the one value, so the minimum-norm solution is a flat curve at
    // that value everywhere, via the B-spline partition-of-unity property),
    // and for any other sparse-relative-to-M channel it's strictly better
    // than 0.
    std::vector<double> mean(C, 0.0);
    for (size_t c = 0; c < C; ++c) {
        for (size_t s = 0; s < S; ++s) mean[c] += values[s][c];
        if (S) mean[c] /= double(S);
    }
    std::vector<std::vector<double>> rhsColumns(C, std::vector<double>(S));
    for (size_t s = 0; s < S; ++s)
        for (size_t c = 0; c < C; ++c) rhsColumns[c][s] = values[s][c] - mean[c];

    auto solved = solveNormalEquations(A, size_t(M), rhsColumns); // [C][M], deviations from mean

    FitResult out;
    out.controlPoints.assign(M, std::vector<double>(C));
    for (int i = 0; i < M; ++i)
        for (size_t c = 0; c < C; ++c) out.controlPoints[i][c] = solved[c][i] + mean[c];

    // Reconstruction error check. Only the row's p+1 entries can be nonzero,
    // and summing them in ascending index order gives the same result as the
    // full dense dot product did (the skipped terms are all exact zeros).
    double maxErr = 0.0;
    for (size_t s = 0; s < S; ++s) {
        const BasisRow& row = A[s];
        for (size_t c = 0; c < C; ++c) {
            double rec = 0.0;
            for (int j = 0; j <= p; ++j) rec += row.v[j] * out.controlPoints[row.base + j][c];
            maxErr = std::max(maxErr, std::fabs(rec - values[s][c]));
        }
    }
    out.maxError = maxErr;
    return out;
}

// Picks a small M in [minM, maxM] with reconstruction error <= tol.
// Each individual fit is O(S log M + M) (sparse basis rows, banded solve),
// but a naive linear scan of every M from minM..maxM is still O(maxM)
// *fits* -- for an animation with an unusually large number of keyframes
// (large maxM), this combination can make one single channel's search
// take a long time. Exponential search for a working upper
// bound, then binary search within it, cuts this to O(log maxM) fits
// instead. maxM is the sample count unless the caller sets
// Options::maxControlPoints; with sparse basis rows each fit is linear in
// S and M, so even the largest attempt stays cheap.
inline FitResult fitBSplineAdaptive(const std::vector<double>& ts,
                                     const std::vector<std::vector<double>>& values,
                                     double tolerance, int minM, int maxM) {
    if (maxM <= minM) return fitBSpline(ts, values, maxM);

    FitResult atMin = fitBSpline(ts, values, minM);
    if (atMin.maxError <= tolerance) return atMin;

    // Exponential search: double M until tolerance is met or we hit maxM.
    int lo = minM;
    int hi = minM;
    FitResult hiResult = atMin;
    while (hiResult.maxError > tolerance && hi < maxM) {
        lo = hi;
        hi = std::min(maxM, hi < 1 ? minM + 1 : hi * 2);
        hiResult = fitBSpline(ts, values, hi);
    }
    if (hiResult.maxError > tolerance) return hiResult; // hit maxM without meeting tolerance

    // Binary search the smallest M in (lo, hi] that still meets tolerance.
    FitResult best = hiResult;
    while (hi - lo > 1) {
        int mid = lo + (hi - lo) / 2;
        FitResult midResult = fitBSpline(ts, values, mid);
        if (midResult.maxError <= tolerance) { hi = mid; best = midResult; }
        else lo = mid;
    }
    return best;
}

} // namespace bspline
