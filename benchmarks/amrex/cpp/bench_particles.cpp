// AMReX side of the patchwork vs AMReX marker benchmark (B6).
//
// The hierarchy of bench_solve (n_cell base cells per axis, each finer level over the
// middle half of the one below, boxes of at most 64 cells), `count` markers drawn from
// a Gaussian of width 0.15 around the centre (clipped to the domain), unit charges.
// Times (median of `repeats`):
// - migrate: move every marker by a small random step, then Redistribute (find the
//   level, box and rank of each marker and send it there);
// - deposit: cloud-in-cell charge onto a nodal field of each marker's level
//   (ParticleToMesh), then SumBoundary (the shared nodes and ghosts added);
// - gather: FillBoundary, then each marker reads the nodal field of its level
//   (MeshToParticle, the transposed weights);
// - sort: SortParticlesByCell.
// Prints one JSON line.

#include <AMReX.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParticleInterpolators.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_Random.H>

#include <algorithm>
#include <chrono>
#include <cmath>
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

// real components: 0 charge, 1 the gathered field
using Container = AmrParticleContainer<2, 0>;
using PType = Container::ParticleType;

}  // namespace

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    {
        ParmParse pp;
        int n_cell = 64, nlevels = 3, repeats = 5;
        Long count = 100000;
        pp.query("n_cell", n_cell);
        pp.query("nlevels", nlevels);
        pp.query("count", count);
        pp.query("repeats", repeats);

        RealBox rb({AMREX_D_DECL(0., 0., 0.)}, {AMREX_D_DECL(1., 1., 1.)});
        Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(0, 0, 0)};
        Vector<Geometry> geom(nlevels);
        Vector<BoxArray> grids(nlevels);
        Vector<DistributionMapping> dmap(nlevels);
        Vector<int> ratio(nlevels - 1, 2);
        Box domain(IntVect(0), IntVect(n_cell - 1)), patch = domain;
        for (int lev = 0; lev < nlevels; ++lev) {
            geom[lev].define(domain, rb, 0, periodic);
            grids[lev].define(patch);
            grids[lev].maxSize(64);
            dmap[lev].define(grids[lev]);
            patch.grow(-patch.length(0) / 4);
            patch.refine(2);
            domain.refine(2);
        }
        Container pc(geom, dmap, grids, ratio);

        // the markers, all made on rank 0 in its first box of level 0, then sent
        // where they belong
        if (ParallelDescriptor::IOProcessor()) {
            MFIter mfi(pc.ParticleBoxArray(0), pc.ParticleDistributionMap(0));
            auto& tile = pc.DefineAndReturnParticleTile(0, mfi.index(), mfi.LocalTileIndex());
            std::mt19937_64 rng(1);
            std::normal_distribution<Real> normal(0.5, 0.15);
            for (Long m = 0; m < count; ++m) {
                PType p;
                p.id() = PType::NextID();
                p.cpu() = ParallelDescriptor::MyProc();
                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    p.pos(d) = std::clamp(normal(rng), Real(1e-6), Real(1 - 1e-6));
                }
                p.rdata(0) = 1.0;
                p.rdata(1) = 0.0;
                tile.push_back(p);
            }
        }
        pc.Redistribute();

        Vector<MultiFab> rho(nlevels);
        for (int lev = 0; lev < nlevels; ++lev) {
            rho[lev].define(amrex::convert(grids[lev], IntVect::TheNodeVector()), dmap[lev], 1, 1);
        }

        std::vector<double> migrate, deposit, gather, sort;
        for (int r = 0; r <= repeats; ++r) {
            // a small random step, then Redistribute
            for (int lev = 0; lev < nlevels; ++lev) {
                for (auto pti = pc.MakeMFIter(lev); pti.isValid(); ++pti) {
                    auto& tile = pc.ParticlesAt(lev, pti);
                    auto* ps = tile.GetArrayOfStructs()().data();
                    const Long np = tile.numParticles();
                    amrex::ParallelForRNG(np, [=](Long i, RandomEngine const& engine) noexcept {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                            Real x = ps[i].pos(d) + 0.002 * (amrex::Random(engine) - 0.5);
                            ps[i].pos(d) = std::clamp(x, Real(1e-6), Real(1 - 1e-6));
                        }
                    });
                }
            }
            double t0 = now();
            pc.Redistribute();
            double t1 = now();
            for (int lev = 0; lev < nlevels; ++lev) {
                rho[lev].setVal(0.0);
                amrex::ParticleToMesh(
                    pc, rho[lev], lev,
                    [=] AMREX_GPU_DEVICE(const PType& p, Array4<Real> const& a,
                                         GpuArray<Real, AMREX_SPACEDIM> const& plo,
                                         GpuArray<Real, AMREX_SPACEDIM> const& dxi) noexcept {
                        ParticleInterpolator::Linear interp(p, plo, dxi);
                        interp.ParticleToMesh(p, a, 0, 0, 1,
                                              [=] AMREX_GPU_DEVICE(const PType& q, int) {
                                                  return q.rdata(0);
                                              });
                    });
                rho[lev].SumBoundary(geom[lev].periodicity());
            }
            double t2 = now();
            for (int lev = 0; lev < nlevels; ++lev) {
                rho[lev].FillBoundary(geom[lev].periodicity());
                amrex::MeshToParticle(
                    pc, rho[lev], lev,
                    [=] AMREX_GPU_DEVICE(PType& p, amrex::Array4<const Real> const& a,
                                         GpuArray<Real, AMREX_SPACEDIM> const& plo,
                                         GpuArray<Real, AMREX_SPACEDIM> const& dxi) noexcept {
                        ParticleInterpolator::Linear interp(p, plo, dxi);
                        interp.MeshToParticle(
                            p, a, 0, 1, 1,
                            [=] AMREX_GPU_DEVICE(const Array4<const Real>& arr, int i, int j,
                                                 int k, int comp) { return arr(i, j, k, comp); },
                            [=] AMREX_GPU_DEVICE(PType& q, int, Real val) { q.rdata(1) = val; });
                    });
            }
            double t3 = now();
            pc.SortParticlesByCell();
            double t4 = now();
            if (r) {
                migrate.push_back(t1 - t0);
                deposit.push_back(t2 - t1);
                gather.push_back(t3 - t2);
                sort.push_back(t4 - t3);
            }
        }
        Real charge = 0.0;
        for (int lev = 0; lev < nlevels; ++lev) charge += rho[lev].sum(0, true);
        ParallelDescriptor::ReduceRealSum(charge);

        amrex::Print() << "{\"code\": \"amrex\", \"ndim\": " << AMREX_SPACEDIM
                       << ", \"n_cell\": " << n_cell << ", \"nlevels\": " << nlevels
                       << ", \"ranks\": " << ParallelDescriptor::NProcs()
                       << ", \"markers\": " << pc.TotalNumberOfParticles()
                       << ", \"migrate_s\": " << median(migrate)
                       << ", \"deposit_s\": " << median(deposit)
                       << ", \"gather_s\": " << median(gather)
                       << ", \"sort_s\": " << median(sort) << ", \"charge\": " << charge
                       << "}\n";
    }
    amrex::Finalize();
}
