// AMReX side of the patchwork vs AMReX mesh benchmarks (B4, B5).
//
// mode=halo (B4): two levels, the base of n_cell cells per axis (boxes of at most 64),
//   the fine level over the middle half, cut into boxes of `tile` fine cells per axis.
//   A cell field with one ghost cell; times (median of `repeats`) of filling the ghosts
//   (FillBoundary on the base, FillPatchTwoLevels with bilinear interpolation on the
//   fine level) and of average_down.
// mode=regrid (B5, 2D): the moving Gaussian charge of patchwork's
//   examples/regrid_moving_charge.py on a 32 x 32 base, cell-centered: per step, tag
//   the cells whose density differs from a neighbour's by more than `threshold`, make
//   new grids (Berger-Rigoutsos, grid_eff 0.7, one buffer cell, proper nesting 1,
//   max_grid_size 32) and fill the density on them from the old grids (FillPatch).
//
// Prints one JSON line.

#include <AMReX.H>
#include <AMReX_AmrMesh.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_TagBox.H>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

using namespace amrex;

namespace {

double now() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

constexpr Real width = 0.002;
constexpr Real pi = 3.14159265358979323846;

AMREX_GPU_HOST_DEVICE Real density(Real x, Real y, Real t) {
    const Real cx = 0.25 + 0.5 * t, cy = 0.3 + 0.4 * t;
    return std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / width) / (pi * width);
}

void set_density(MultiFab& mf, const Geometry& geom, Real t) {
    const auto dx = geom.CellSizeArray();
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        auto a = mf.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=](int i, int j, int k) noexcept {
            amrex::ignore_unused(k);
            a(i, j, k) = density((i + 0.5) * dx[0], (j + 0.5) * dx[1], t);
        });
    }
}

// An AmrMesh whose ErrorEst tags where the density differs from a neighbour's by
// more than the threshold (the rule of patchwork's gradient_tags for cell fields).
class Moving : public AmrMesh {
  public:
    Moving(const RealBox& rb, int max_level, const Vector<int>& n_cell, Real threshold)
        : AmrMesh(&rb, max_level, n_cell, 0, Vector<IntVect>(max_level, IntVect(2))),
          threshold(threshold) {}

    Real t = 0.0;
    Real threshold;
    Vector<MultiFab> rho;

    void ErrorEst(int lev, TagBoxArray& tags, Real, int) override {
        MultiFab& r = rho[lev];
        r.FillBoundary(Geom(lev).periodicity());
        const Real thr = threshold;
        const Box domain = Geom(lev).Domain();
        for (MFIter mfi(r); mfi.isValid(); ++mfi) {
            auto a = r.const_array(mfi);
            auto tag = tags.array(mfi);
            const Box& bx = mfi.validbox();
            amrex::ParallelFor(bx, [=](int i, int j, int k) noexcept {
                Real c = a(i, j, k);
                bool hit = false;
                if (i > domain.smallEnd(0) && std::abs(c - a(i - 1, j, k)) > thr) hit = true;
                if (i < domain.bigEnd(0) && std::abs(c - a(i + 1, j, k)) > thr) hit = true;
                if (j > domain.smallEnd(1) && std::abs(c - a(i, j - 1, k)) > thr) hit = true;
                if (j < domain.bigEnd(1) && std::abs(c - a(i, j + 1, k)) > thr) hit = true;
                if (hit) tag(i, j, k) = TagBox::SET;
            });
        }
    }

    // Unused: the grids are made with MakeNewGrids directly.
    void MakeNewLevelFromScratch(int, Real, const BoxArray&, const DistributionMapping&) override {}
};

PhysBCFunctNoOp noop;

int halo_mode(int n_cell, int tile, int repeats) {
    RealBox rb({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
    Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(0, 0, 0)};
    Box domain(IntVect(0), IntVect(n_cell - 1));
    Geometry g0(domain, rb, 0, periodic), g1(amrex::refine(domain, 2), rb, 0, periodic);
    BoxArray b0(domain);
    b0.maxSize(64);
    Box fine = amrex::refine(amrex::grow(domain, -n_cell / 4), 2);
    BoxArray b1(fine);
    b1.maxSize(tile);
    DistributionMapping d0(b0), d1(b1);
    MultiFab u0(b0, d0, 1, 1), u1(b1, d1, 1, 1);
    u0.setVal(1.0);
    u1.setVal(2.0);
    Vector<BCRec> bcs(1);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        bcs[0].setLo(d, BCType::ext_dir);
        bcs[0].setHi(d, BCType::ext_dir);
    }
    Interpolater* interp = &cell_bilinear_interp;
    std::vector<double> fill, restrict;
    for (int r = 0; r <= repeats; ++r) {
        double t0 = now();
        u0.FillBoundary(g0.periodicity());
        amrex::FillPatchTwoLevels(u1, 0.0, {&u0}, {0.0}, {&u1}, {0.0}, 0, 0, 1, g0, g1,
                                  noop, 0, noop, 0, IntVect(2), interp, bcs, 0);
        double t1 = now();
        amrex::average_down(u1, u0, g1, g0, 0, 1, 2);
        double t2 = now();
        if (r) {
            fill.push_back(t1 - t0);
            restrict.push_back(t2 - t1);
        }
    }
    amrex::Print() << "{\"code\": \"amrex\", \"mode\": \"halo\", \"ndim\": " << AMREX_SPACEDIM
                   << ", \"n_cell\": " << n_cell << ", \"tile\": " << tile
                   << ", \"patches\": " << b0.size() + b1.size()
                   << ", \"fine_patches\": " << b1.size() << ", \"cells\": "
                   << b0.numPts() + b1.numPts() << ", \"fill_s\": " << median(fill)
                   << ", \"restrict_s\": " << median(restrict) << "}\n";
    return 0;
}

int regrid_mode(int steps, int max_level, Real threshold, int blocking_factor) {
#if AMREX_SPACEDIM == 2
    {
        ParmParse pp("amr");
        pp.add("blocking_factor", blocking_factor);
        pp.add("max_grid_size", 32);
        pp.add("grid_eff", 0.7);
        pp.add("n_error_buf", 1);
        pp.add("n_proper", 1);
    }
    RealBox rb({0., 0.}, {1., 1.});
    Moving mesh(rb, max_level, {32, 32}, threshold);
    const Box domain = mesh.Geom(0).Domain();
    BoxArray ba(domain);
    ba.maxSize(32);
    mesh.SetBoxArray(0, ba);
    mesh.SetDistributionMap(0, DistributionMapping(ba));
    mesh.SetFinestLevel(0);
    mesh.rho.resize(max_level + 1);
    mesh.rho[0].define(ba, mesh.DistributionMap(0), 1, 1);
    set_density(mesh.rho[0], mesh.Geom(0), 0.0);

    Vector<BCRec> bcs(1);
    for (int d = 0; d < 2; ++d) {
        bcs[0].setLo(d, BCType::foextrap);
        bcs[0].setHi(d, BCType::foextrap);
    }
    Interpolater* interp = &cell_bilinear_interp;

    // one regrid: new grids from the tags, the density moved onto them
    auto regrid = [&](Real t) {
        int new_finest = 0;
        Vector<BoxArray> grids(max_level + 1);
        mesh.MakeNewGrids(0, t, new_finest, grids);
        Vector<MultiFab> moved(max_level + 1);
        moved[0] = std::move(mesh.rho[0]);
        for (int lev = 1; lev <= new_finest; ++lev) {
            DistributionMapping dm(grids[lev]);
            moved[lev].define(grids[lev], dm, 1, 1);
            if (lev <= mesh.finestLevel()) {
                amrex::FillPatchTwoLevels(moved[lev], t, {&moved[lev - 1]}, {t},
                                          {&mesh.rho[lev]}, {t}, 0, 0, 1,
                                          mesh.Geom(lev - 1), mesh.Geom(lev), noop, 0, noop,
                                          0, IntVect(2), interp, bcs, 0);
            } else {
                amrex::InterpFromCoarseLevel(moved[lev], t, moved[lev - 1], 0, 0, 1,
                                             mesh.Geom(lev - 1), mesh.Geom(lev), noop, 0,
                                             noop, 0, IntVect(2), interp, bcs, 0);
            }
            mesh.SetBoxArray(lev, grids[lev]);
            mesh.SetDistributionMap(lev, dm);
        }
        for (int lev = new_finest + 1; lev <= mesh.finestLevel(); ++lev) {
            mesh.ClearBoxArray(lev);
            mesh.ClearDistributionMap(lev);
        }
        mesh.SetFinestLevel(new_finest);
        for (int lev = 0; lev <= max_level; ++lev) mesh.rho[lev] = std::move(moved[lev]);
    };
    auto deposit = [&](Real t) {
        for (int lev = 0; lev <= mesh.finestLevel(); ++lev)
            set_density(mesh.rho[lev], mesh.Geom(lev), t);
    };

    // deepen at t = 0 until the levels stop growing, as the patchwork example does
    for (int k = 0; k < max_level; ++k) {
        regrid(0.0);
        deposit(0.0);
    }
    std::vector<double> per_step;
    long patches = 0, cells = 0;
    int levels = 0;
    for (int step = 1; step <= steps; ++step) {
        const Real t = Real(step) / steps;
        deposit(t);
        double t0 = now();
        regrid(t);
        per_step.push_back(now() - t0);
        deposit(t);
        patches = 0;
        cells = 0;
        for (int lev = 0; lev <= mesh.finestLevel(); ++lev) {
            patches += mesh.boxArray(lev).size();
            cells += mesh.boxArray(lev).numPts();
        }
        levels = mesh.finestLevel() + 1;
    }
    double total = 0;
    for (double x : per_step) total += x;
    amrex::Print() << "{\"code\": \"amrex\", \"mode\": \"regrid\", \"steps\": " << steps
                   << ", \"blocking_factor\": " << blocking_factor
                   << ", \"regrid_s\": " << total / steps << ", \"levels\": " << levels
                   << ", \"patches\": " << patches << ", \"cells\": " << cells << "}\n";
#else
    amrex::ignore_unused(steps, max_level, threshold, blocking_factor);
    amrex::Abort("regrid mode is 2D");
#endif
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;
        std::string mode = "halo";
        int n_cell = 128, tile = 32, repeats = 20, steps = 20, max_level = 4,
            blocking_factor = 2;
        Real threshold = 10.0;
        pp.query("mode", mode);
        pp.query("n_cell", n_cell);
        pp.query("tile", tile);
        pp.query("repeats", repeats);
        pp.query("steps", steps);
        pp.query("max_level", max_level);
        pp.query("threshold", threshold);
        pp.query("blocking_factor", blocking_factor);
        if (mode == "halo") {
            halo_mode(n_cell, tile, repeats);
        } else {
            regrid_mode(steps, max_level, threshold, blocking_factor);
        }
    }
    amrex::Finalize();
}
