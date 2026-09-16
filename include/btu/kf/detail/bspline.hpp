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

// Cox-de Boor recursive basis function N_{i,p}(t) via the standard
// dynamic-programming table (O(p^2) per evaluation).
inline void basisRow(const std::vector<double>& knots, int M, int p, double t,
                      std::vector<double>& outN /* size M */) {
    outN.assign(M, 0.0);
    // find knot span
    int n = M - 1; // highest control point index
    if (t >= knots[n + 1]) t = std::nextafter(knots[n + 1], -1.0);
    if (t < knots[p]) t = knots[p];

    // Table of basis values via triangular recurrence.
    std::vector<double> N(p + 1, 0.0);
    int span = p;
    for (int i = p; i <= n; ++i) {
        if (t >= knots[i] && t < knots[i + 1]) { span = i; break; }
        if (i == n) span = n; // last span is closed on the right
    }

    std::vector<double> left(p + 1), right(p + 1);
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
    for (int j = 0; j <= p; ++j) {
        int idx = span - p + j;
        if (idx >= 0 && idx < M) outN[idx] = N[j];
    }
}

// Solve (A^T A) x = A^T b for several right-hand-side columns at once via
// LU decomposition with partial pivoting on the (small, dense) MxM normal
// matrix. `rhs` is stored column-major: rhs[col][row].
inline std::vector<std::vector<double>> solveNormalEquations(
    const std::vector<std::vector<double>>& A,             // S x M
    const std::vector<std::vector<double>>& rhsColumns) {  // C columns, each length S
    size_t S = A.size();
    size_t M = S ? A[0].size() : 0;
    size_t C = rhsColumns.size();

    // Build normal matrix ATA (M x M) and ATb (M x C).
    std::vector<std::vector<double>> ATA(M, std::vector<double>(M, 0.0));
    std::vector<std::vector<double>> ATb(M, std::vector<double>(C, 0.0));
    for (size_t s = 0; s < S; ++s) {
        for (size_t i = 0; i < M; ++i) {
            double ai = A[s][i];
            if (ai == 0.0) continue;
            for (size_t j = 0; j < M; ++j) ATA[i][j] += ai * A[s][j];
            for (size_t c = 0; c < C; ++c) ATb[i][c] += ai * rhsColumns[c][s];
        }
    }
    // Tiny Tikhonov regularization for numerical stability on
    // under-determined / degenerate systems (e.g. constant channels).
    for (size_t i = 0; i < M; ++i) ATA[i][i] += 1e-9;

    // Roughness penalty on the control polygon's discrete second
    // derivative (D^T D, D = second-difference operator). Required, not
    // cosmetic: the real engine's NiBSplineBasis::Compute hard-codes a
    // *uniform* clamped knot vector at decode time (confirmed in
    // recon/ida_dump.txt), so the encoder cannot switch to chord-length
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
        for (size_t i = 0; i < M; ++i) diagSum += ATA[i][i];
        double lambda = (diagSum / double(M)) * 1e-3;
        if (lambda > 0.0) {
            for (size_t i = 0; i + 2 < M; ++i) {
                ATA[i][i]     += lambda;
                ATA[i][i+1]   -= 2.0 * lambda;
                ATA[i][i+2]   += lambda;
                ATA[i+1][i]   -= 2.0 * lambda;
                ATA[i+1][i+1] += 4.0 * lambda;
                ATA[i+1][i+2] -= 2.0 * lambda;
                ATA[i+2][i]   += lambda;
                ATA[i+2][i+1] -= 2.0 * lambda;
                ATA[i+2][i+2] += lambda;
            }
        }
    }

    // Gaussian elimination with partial pivoting, augmented with C RHS columns.
    std::vector<std::vector<double>> aug(M, std::vector<double>(M + C));
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) aug[i][j] = ATA[i][j];
        for (size_t c = 0; c < C; ++c) aug[i][M + c] = ATb[i][c];
    }
    for (size_t col = 0; col < M; ++col) {
        size_t piv = col;
        double best = std::fabs(aug[col][col]);
        for (size_t r = col + 1; r < M; ++r) {
            double v = std::fabs(aug[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) std::swap(aug[piv], aug[col]);
        double d = aug[col][col];
        if (std::fabs(d) < 1e-15) d = (d < 0 ? -1e-15 : 1e-15);
        for (size_t r = 0; r < M; ++r) {
            if (r == col) continue;
            double factor = aug[r][col] / d;
            if (factor == 0.0) continue;
            for (size_t c2 = col; c2 < M + C; ++c2) aug[r][c2] -= factor * aug[col][c2];
        }
    }
    std::vector<std::vector<double>> result(C, std::vector<double>(M));
    for (size_t i = 0; i < M; ++i) {
        double d = aug[i][i];
        if (std::fabs(d) < 1e-15) d = (d < 0 ? -1e-15 : 1e-15);
        for (size_t c = 0; c < C; ++c) result[c][i] = aug[i][M + c] / d;
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
    // Confirmed as a real, widespread bug this way: validating a full
    // 14,313-file real batch found 1,784 files with exactly this handle/M
    // mismatch on scale channels specifically (scale is almost always the
    // sparsest channel on a bone), traced back to this clamp silently
    // returning fewer control points than the caller asked for.

    auto knots = clampedUniformKnots(M, p);

    std::vector<std::vector<double>> A(S, std::vector<double>(M));
    for (size_t s = 0; s < S; ++s) basisRow(knots, M, p, ts[s], A[s]);

    size_t C = values.empty() ? 0 : values[0].size();

    // Fit DEVIATIONS from each component's own mean, not the raw values.
    // Why: when M is large relative to S (a sparse channel -- e.g. a
    // 1-key scale track -- sharing a bone's basis block with a much
    // denser channel, forced up to that bone's shared M), most control
    // points have no real constraint from any sample and are set purely
    // by the Tikhonov regularization term (ATA[i][i] += 1e-9 below),
    // which pulls an unconstrained control point toward *zero*. For a
    // scale channel (nominally ~1.0) that means most of the curve
    // decodes near 0 instead of near 1 -- i.e. the character's scale
    // collapses to nothing for most of the clip: a real, confirmed
    // "character is the size of a smurf" bug, found by fit-quality
    // testing on a real 1-key scale track forced to M=356 by its bone's
    // dense translation/rotation channels. The same failure mode exists
    // for rotation (an unconstrained quaternion component droops to 0,
    // which is a degenerate, non-unit quaternion, not identity) and for
    // translation (droops to world origin, not the bone's own position).
    // Centering the fit on each component's mean makes the regularizer's
    // implicit default "hold this channel's own typical value" instead
    // of "hold zero" -- for the degenerate single-key case this is exact
    // (mean == the one value, so the deviation target is all zeros and
    // the minimum-norm solution is a flat curve at that value, everywhere,
    // via the B-spline partition-of-unity property), and for any other
    // sparse-relative-to-M channel it's a strictly better default than 0.
    std::vector<double> mean(C, 0.0);
    for (size_t c = 0; c < C; ++c) {
        for (size_t s = 0; s < S; ++s) mean[c] += values[s][c];
        if (S) mean[c] /= double(S);
    }
    std::vector<std::vector<double>> rhsColumns(C, std::vector<double>(S));
    for (size_t s = 0; s < S; ++s)
        for (size_t c = 0; c < C; ++c) rhsColumns[c][s] = values[s][c] - mean[c];

    auto solved = solveNormalEquations(A, rhsColumns); // [C][M], deviations from mean

    FitResult out;
    out.controlPoints.assign(M, std::vector<double>(C));
    for (int i = 0; i < M; ++i)
        for (size_t c = 0; c < C; ++c) out.controlPoints[i][c] = solved[c][i] + mean[c];

    // Reconstruction error check.
    double maxErr = 0.0;
    std::vector<double> row(M);
    for (size_t s = 0; s < S; ++s) {
        basisRow(knots, M, p, ts[s], row);
        for (size_t c = 0; c < C; ++c) {
            double rec = 0.0;
            for (int i = 0; i < M; ++i) rec += row[i] * out.controlPoints[i][c];
            maxErr = std::max(maxErr, std::fabs(rec - values[s][c]));
        }
    }
    out.maxError = maxErr;
    return out;
}

// Picks a small M in [minM, maxM] with reconstruction error <= tol.
// Each individual fit is O(M^3) (dense Gaussian elimination on the MxM
// normal matrix), so a naive linear scan of every M from minM..maxM is
// O(maxM) *fits*, and the cost of those fits themselves grows cubically
// as M grows -- for an animation with an unusually large number of
// keyframes (large maxM), this combination can make one single channel's
// search take a very long time. Exponential search for a working upper
// bound, then binary search within it, cuts this to O(log maxM) fits
// instead -- found the hard way when one outlier file in a 14,000+ file
// batch made one worker thread hang long enough that the whole run
// looked stuck (the other threads had already finished everything else
// and were just waiting at the join). Callers should also externally cap
// maxM (see Options::maxControlPoints in main.cpp) since even O(log maxM)
// fits doesn't bound the cost of the single largest fit attempted.
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
