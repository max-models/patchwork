// AMReX side of the patchwork vs AMReX solver benchmarks (B1-B3).
//
// mode=nodal: div(grad phi) = rhs on the nodes (MLNodeLaplacian), the hierarchy
//   given by n_cell and nlevels: each finer level covers the middle half of the one
//   below (in cells of the base level: [n/4, 3n/4), then [3n/8, 5n/8), ...).
// mode=abec: -div(a grad u) + c u = f on the cells (MLABecLaplacian), 2D only
//   in the benchmark, a = 1 + 0.5 sin(2 pi x) sin(2 pi y), c = 1.
// The exact solution is a Gaussian exp(-|x - 0.5|^2 / s^2), s = 0.1, zero (to
// 1e-11) on the boundary, which is Dirichlet zero.
//
// Prints one JSON line: build, first and repeated solve times, iterations, the
// max-norm error over all levels and the peak resident memory.

#include <AMReX.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLNodeLaplacian.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_iMultiFab.H>

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

using namespace amrex;

namespace {

constexpr Real s = 0.1;
constexpr Real pi = 3.14159265358979323846;

double now() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double peak_mb() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

AMREX_GPU_HOST_DEVICE Real gaussian(Real x, Real y, Real z, Real c = 0.5) {
    Real r2 = (x - c) * (x - c) + (y - c) * (y - c);
#if AMREX_SPACEDIM == 3
    r2 += (z - c) * (z - c);
#else
    amrex::ignore_unused(z);
#endif
    return std::exp(-r2 / (s * s));
}

// Laplacian of the Gaussian
AMREX_GPU_HOST_DEVICE Real laplacian(Real x, Real y, Real z, Real c = 0.5) {
    Real r2 = (x - c) * (x - c) + (y - c) * (y - c);
#if AMREX_SPACEDIM == 3
    r2 += (z - c) * (z - c);
#endif
    return gaussian(x, y, z, c) * (4.0 * r2 / (s * s * s * s) - 2.0 * AMREX_SPACEDIM / (s * s));
}

AMREX_GPU_HOST_DEVICE Real coefficient(Real x, Real y) {
    return 1.0 + 0.5 * std::sin(2 * pi * x) * std::sin(2 * pi * y);
}

// f = -div(a grad u) + u for the 2D abec case
AMREX_GPU_HOST_DEVICE Real abec_source(Real x, Real y, Real c = 0.5) {
    Real u = gaussian(x, y, 0.0, c);
    Real ux = -2.0 * (x - c) / (s * s) * u;
    Real uy = -2.0 * (y - c) / (s * s) * u;
    Real ax = 0.5 * 2 * pi * std::cos(2 * pi * x) * std::sin(2 * pi * y);
    Real ay = 0.5 * 2 * pi * std::sin(2 * pi * x) * std::cos(2 * pi * y);
    return -(ax * ux + ay * uy + coefficient(x, y) * laplacian(x, y, 0.0, c)) + u;
}

}  // namespace

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;
        std::string mode = "nodal";
        int n_cell = 64, nlevels = 3, max_grid_size = 64, repeats = 5, verbose = 0;
        Real reltol = 1.e-10;
        pp.query("mode", mode);
        pp.query("n_cell", n_cell);
        pp.query("nlevels", nlevels);
        pp.query("max_grid_size", max_grid_size);
        pp.query("repeats", repeats);
        pp.query("reltol", reltol);
        pp.query("verbose", verbose);
        const bool nodal = mode == "nodal";

        double t0 = now();
        RealBox rb({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
        Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
        Vector<Geometry> geom(nlevels);
        Vector<BoxArray> grids(nlevels);
        Vector<DistributionMapping> dmap(nlevels);
        Box domain(IntVect(0), IntVect(n_cell - 1));
        Box patch = domain;
        for (int lev = 0; lev < nlevels; ++lev) {
            geom[lev].define(domain, rb, 0, is_periodic);
            grids[lev].define(patch);
            grids[lev].maxSize(max_grid_size);
            dmap[lev].define(grids[lev]);
            // the next level covers the middle half of this one's patch
            const int quarter = patch.length(0) / 4;
            patch.grow(-quarter);
            patch.refine(2);
            domain.refine(2);
        }

        // a new source every solve (no warm start), the last one centred at 0.5
        auto centre = [&](int r) { return 0.5 + 0.04 * ((repeats - r) % 3); };
        Vector<MultiFab> solution(nlevels), rhs(nlevels), exact(nlevels), acoef(nlevels),
            sigma(nlevels);
        Vector<Vector<MultiFab>> rhss(3);
        Vector<Array<MultiFab, AMREX_SPACEDIM>> bcoef(nlevels);
        for (int lev = 0; lev < nlevels; ++lev) {
            const BoxArray ba = nodal ? amrex::convert(grids[lev], IntVect::TheNodeVector())
                                      : grids[lev];
            const int ghosts = nodal ? 0 : 1;
            solution[lev].define(ba, dmap[lev], 1, ghosts);
            rhs[lev].define(ba, dmap[lev], 1, 0);
            exact[lev].define(ba, dmap[lev], 1, 0);
            for (int k = 0; k < 3; ++k) {
                rhss[k].resize(nlevels);
                rhss[k][lev].define(ba, dmap[lev], 1, 0);
            }
            solution[lev].setVal(0.0);
            const auto dx = geom[lev].CellSizeArray();
            const Real offset = nodal ? 0.0 : 0.5;
            for (MFIter mfi(rhs[lev]); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto ex = exact[lev].array(mfi);
                auto rh = rhs[lev].array(mfi);
                amrex::ParallelFor(bx, [=](int i, int j, int k) noexcept {
                    Real x = (i + offset) * dx[0];
                    Real y = (j + offset) * dx[1];
#if AMREX_SPACEDIM == 3
                    Real z = (k + offset) * dx[2];
#else
                    Real z = 0.0;
#endif
                    ex(i, j, k) = gaussian(x, y, z);
                    rh(i, j, k) = nodal ? laplacian(x, y, z) : abec_source(x, y);
                });
                for (int m = 0; m < 3; ++m) {
                    auto rm = rhss[m][lev].array(mfi);
                    const Real c = 0.5 + 0.04 * m;
                    amrex::ParallelFor(bx, [=](int i, int j, int k) noexcept {
                        Real x = (i + offset) * dx[0];
                        Real y = (j + offset) * dx[1];
#if AMREX_SPACEDIM == 3
                        Real z = (k + offset) * dx[2];
#else
                        Real z = 0.0;
#endif
                        rm(i, j, k) = nodal ? laplacian(x, y, z, c) : abec_source(x, y, c);
                    });
                }
            }
            if (nodal) {
                sigma[lev].define(grids[lev], dmap[lev], 1, 0);
                sigma[lev].setVal(1.0);
            } else {
                acoef[lev].define(grids[lev], dmap[lev], 1, 0);
                acoef[lev].setVal(1.0);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    const BoxArray fba =
                        amrex::convert(grids[lev], IntVect::TheDimensionVector(d));
                    bcoef[lev][d].define(fba, dmap[lev], 1, 0);
                    for (MFIter mfi(bcoef[lev][d]); mfi.isValid(); ++mfi) {
                        auto b = bcoef[lev][d].array(mfi);
                        amrex::ParallelFor(mfi.validbox(), [=](int i, int j, int k) noexcept {
                            Real x = (i + (d == 0 ? 0.0 : 0.5)) * dx[0];
                            Real y = (j + (d == 1 ? 0.0 : 0.5)) * dx[1];
                            amrex::ignore_unused(k);
                            b(i, j, k) = coefficient(x, y);
                        });
                    }
                }
            }
        }

        LPInfo info;
        std::unique_ptr<MLLinOp> linop;
        Array<LinOpBCType, AMREX_SPACEDIM> dirichlet{
            AMREX_D_DECL(LinOpBCType::Dirichlet, LinOpBCType::Dirichlet, LinOpBCType::Dirichlet)};
        double t_data = now() - t0;
        t0 = now();
        if (nodal) {
            auto op = std::make_unique<MLNodeLaplacian>(geom, grids, dmap, info);
            op->setDomainBC(dirichlet, dirichlet);
            for (int lev = 0; lev < nlevels; ++lev) op->setSigma(lev, sigma[lev]);
            linop = std::move(op);
        } else {
            auto op = std::make_unique<MLABecLaplacian>(geom, grids, dmap, info);
            op->setDomainBC(dirichlet, dirichlet);
            for (int lev = 0; lev < nlevels; ++lev) op->setLevelBC(lev, &solution[lev]);
            op->setScalars(1.0, 1.0);
            for (int lev = 0; lev < nlevels; ++lev) {
                op->setACoeffs(lev, acoef[lev]);
                op->setBCoeffs(lev, amrex::GetArrOfConstPtrs(bcoef[lev]));
            }
            linop = std::move(op);
        }
        MLMG mlmg(*linop);
        mlmg.setVerbose(verbose);
        mlmg.setMaxIter(200);
        double t_build = now() - t0;

        std::vector<double> times;
        int iterations = 0;
        for (int r = 0; r <= repeats; ++r) {
            for (int lev = 0; lev < nlevels; ++lev) solution[lev].setVal(0.0);
            t0 = now();
            const int m = static_cast<int>(std::lround((centre(r) - 0.5) / 0.04));
            mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhss[m]), reltol, 0.0);
            times.push_back(now() - t0);
            iterations = mlmg.getNumIters();
        }
        const double first = times[0];
        std::vector<double> rest(times.begin() + 1, times.end());
        std::sort(rest.begin(), rest.end());
        const double median = rest.empty() ? first : rest[rest.size() / 2];

        Real error = 0.0;
        long unknowns = 0;
        for (int lev = 0; lev < nlevels; ++lev) {
            MultiFab diff(rhs[lev].boxArray(), dmap[lev], 1, 0);
            MultiFab::Copy(diff, solution[lev], 0, 0, 1, 0);
            MultiFab::Subtract(diff, exact[lev], 0, 0, 1, 0);
            if (!nodal && lev + 1 < nlevels) {
                // covered cells hold the mean of the fine cells, not a point value
                const iMultiFab mask = amrex::makeFineMask(diff, grids[lev + 1], IntVect(2));
                for (MFIter mfi(diff); mfi.isValid(); ++mfi) {
                    auto d = diff.array(mfi);
                    auto m = mask.const_array(mfi);
                    amrex::ParallelFor(mfi.validbox(), [=](int i, int j, int k) noexcept {
                        if (m(i, j, k)) d(i, j, k) = 0.0;
                    });
                }
            }
            error = std::max(error, diff.norm0());
            unknowns += rhs[lev].boxArray().numPts();
        }

        amrex::Print() << "{\"code\": \"amrex\", \"mode\": \"" << mode
                       << "\", \"ndim\": " << AMREX_SPACEDIM << ", \"n_cell\": " << n_cell
                       << ", \"nlevels\": " << nlevels << ", \"ranks\": "
                       << ParallelDescriptor::NProcs() << ", \"points\": " << unknowns
                       << ", \"data_s\": " << t_data << ", \"build_s\": " << t_build
                       << ", \"first_solve_s\": " << first << ", \"solve_s\": " << median
                       << ", \"iterations\": " << iterations << ", \"error\": " << error
                       << ", \"peak_mb\": " << peak_mb() << "}\n";
    }
    amrex::Finalize();
}
