// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Stencils.h>
#include <openvdb/tools/LevelSetAdvect.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTracker.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>

namespace {

/// Has the same 8^3 leaves as openvdb::FloatGrid but is a distinct type, because its internal
/// nodes differ. The specialization below keeps it on the math:: stencils, which makes it the
/// reference that the leaf stencil is compared against.
using ReferenceGrid = openvdb::Grid<openvdb::tree::Tree4<float, 4, 5, 3>::Type>;

} // namespace

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {
namespace lstrack {
template<> struct UseLeafStarStencil<ReferenceGrid>
{
    static constexpr bool value = false;
};
} // namespace lstrack
} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


class TestStencils : public ::testing::Test
{
};

TEST_F(TestStencils, testMinMax)
{
    using namespace openvdb;

    Int32Grid grid;
    grid.tree().setValue(math::Coord(0,0,0), -3);
    grid.tree().setValue(math::Coord(0,0,1), -2);
    grid.tree().setValue(math::Coord(0,1,0), -1);
    grid.tree().setValue(math::Coord(1,0,0),  0);
    grid.tree().setValue(math::Coord(1,1,0),  1);
    grid.tree().setValue(math::Coord(0,1,1),  2);
    grid.tree().setValue(math::Coord(1,0,1),  3);
    grid.tree().setValue(math::Coord(1,1,1),  4);
    math::BoxStencil<Int32Grid> stencil(grid);

    stencil.moveTo(Coord(0,0,0));
    EXPECT_EQ(stencil.min(), -3);
    EXPECT_EQ(stencil.max(), 4);

    stencil.moveTo(Coord(1,1,1));
    EXPECT_EQ(stencil.min(), 0);
    EXPECT_EQ(stencil.max(), 4);

    stencil.moveTo(Coord(0,0,1));
    EXPECT_EQ(stencil.min(), -2);
    EXPECT_EQ(stencil.max(),  4);
}

namespace {

using namespace openvdb;

template<typename ValueT>
bool bitsDiffer(const ValueT& a, const ValueT& b)
{
    return std::memcmp(&a, &b, sizeof(ValueT)) != 0;
}

/// Grid of random values whose leaves have every kind of face neighbor: another leaf, the
/// background, and active and inactive tiles. It spans negative coordinates.
template<typename GridT>
typename GridT::Ptr makeRandomGrid()
{
    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;
    using ValueT = typename GridT::ValueType;
    const int leafDim = int(LeafT::DIM);
    const int nodeDim = int(TreeT::RootNodeType::ChildNodeType::ChildNodeType::DIM);

    std::mt19937 rng(20261102);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    std::bernoulli_distribution coin(0.5), skip(0.25);

    typename GridT::Ptr grid = GridT::create(ValueT(7));
    TreeT& tree = grid->tree();

    // All leaves are in the node at (-nodeDim)^3, which has a tile on each of its six faces.
    const Coord node(-nodeDim);
    for (int axis = 0, n = 0; axis < 3; ++axis) {
        for (int side = -1; side <= 1; side += 2, ++n) {
            Coord tile = node;
            tile[axis] += side * nodeDim;
            tree.addTile(/*level=*/1, tile, ValueT(11 + n), /*active=*/side < 0);
        }
    }

    const auto addLeaf = [&](const Coord& origin) {
        LeafT* leaf = tree.touchLeaf(origin);
        for (Index n = 0; n < LeafT::SIZE; ++n) {
            const ValueT value = ValueT(random(rng));
            if (coin(rng)) leaf->setValueOn(n, value);
            else           leaf->setValueOff(n, value);
        }
    };

    // Two blocks of leaves, with holes, touching opposite corners of the node (so the tiles).
    const int blockDim = 4 * leafDim;
    for (const Coord& corner : { node, node.offsetBy(nodeDim - blockDim) }) {
        for (int i = 0; i < blockDim; i += leafDim) {
            for (int j = 0; j < blockDim; j += leafDim) {
                for (int k = 0; k < blockDim; k += leafDim) {
                    if (!skip(rng)) addLeaf(corner.offsetBy(i, j, k));
                }
            }
        }
    }
    // A leaf with only the background around it.
    addLeaf(node.offsetBy(nodeDim / 2));
    return grid;
}

/// Number of the axis offsets up to distance D where getValue<i, j, k>() differs.
template<int D, typename StencilA, typename StencilB>
Index64 offsetMismatches(const StencilA& a, const StencilB& b)
{
    Index64 n = 0;
    n += bitsDiffer(a.template getValue< D, 0, 0>(), b.template getValue< D, 0, 0>());
    n += bitsDiffer(a.template getValue<-D, 0, 0>(), b.template getValue<-D, 0, 0>());
    n += bitsDiffer(a.template getValue< 0, D, 0>(), b.template getValue< 0, D, 0>());
    n += bitsDiffer(a.template getValue< 0,-D, 0>(), b.template getValue< 0,-D, 0>());
    n += bitsDiffer(a.template getValue< 0, 0, D>(), b.template getValue< 0, 0, D>());
    n += bitsDiffer(a.template getValue< 0, 0,-D>(), b.template getValue< 0, 0,-D>());
    if constexpr (D > 1) n += offsetMismatches<D - 1>(a, b);
    return n;
}

/// Assert that the leaf stencil of the given radius replaces @a MathStencilT, and that both
/// read identical values at every voxel of every leaf of @a grid, active or not.
template<typename GridT, typename MathStencilT, int Radius>
void checkGather(const GridT& grid)
{
    using StarT = tools::lstrack::LeafStarStencil<GridT, Radius>;
    static_assert(std::is_same<typename tools::lstrack::LeafStencil<GridT, MathStencilT>::Type,
        StarT>::value, "the selector must replace this stencil");

    const auto mismatches = [](const StarT& star, const MathStencilT& math) {
        Index64 n = star.getCenterCoord() != math.getCenterCoord();
        for (unsigned pos = 0; pos < unsigned(StarT::SIZE); ++pos) {
            n += bitsDiffer(star.getValue(pos), math.getValue(pos));
        }
        return n + bitsDiffer(star.template getValue<0, 0, 0>(), math.template getValue<0, 0, 0>())
            + offsetMismatches<Radius>(star, math);
    };

    MathStencilT math(grid);
    StarT star(grid);
    Index64 voxels = 0, total = 0;
    for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
        star.setLeaf(*leaf);
        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel, ++voxels) {
            math.moveTo(voxel);
            star.moveTo(voxel); // the iterator overload
            total += mismatches(star, math);
            star.moveTo(voxel.getCoord(), *voxel); // the coordinate overload
            total += mismatches(star, math);
        }
    }
    EXPECT_GT(voxels, Index64(0));
    EXPECT_EQ(total, Index64(0));
}

template<typename GridT>
void checkAllGathers()
{
    const typename GridT::Ptr grid = makeRandomGrid<GridT>();
    checkGather<GridT, math::SevenPointStencil<GridT>, 1>(*grid);
    checkGather<GridT, math::ThirteenPointStencil<GridT>, 2>(*grid);
    checkGather<GridT, math::NineteenPointStencil<GridT>, 3>(*grid);
}

/// A sphere that is no longer a signed distance field, so the tracker has work to do.
template<typename GridT>
typename GridT::Ptr makeDistortedSphere()
{
    typename GridT::Ptr grid = tools::createLevelSetSphere<GridT>(
        0.15f, Vec3f(0.35f), 1.0f / 64.0f);
    for (auto voxel = grid->beginValueOn(); voxel; ++voxel) {
        const Coord c = voxel.getCoord();
        const double distortion =
            std::sin(0.7 * c.x()) * std::cos(0.9 * c.y()) * std::sin(1.1 * c.z());
        voxel.setValue(float(double(*voxel) * (1.0 + 0.2 * distortion)));
    }
    return grid;
}

/// Number of voxels in the leaves of @a a whose active state differs in @a b, or whose value
/// differs by more than 1e-5 of a voxel. Exact equality is not expected here: compilers can
/// fuse multiply-adds differently in the two instantiations, which changes the last bits.
template<typename GridA, typename GridB>
Index64 voxelMismatches(const GridA& a, const GridB& b)
{
    const double tolerance = 1e-5 * a.voxelSize()[0];
    Index64 n = 0;
    auto accessor = b.getConstAccessor();
    for (auto leaf = a.tree().cbeginLeaf(); leaf; ++leaf) {
        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel) {
            const Coord c = voxel.getCoord();
            n += std::abs(double(*voxel) - double(accessor.getValue(c))) > tolerance
                || voxel.isValueOn() != accessor.isValueOn(c);
        }
    }
    return n;
}

template<typename GridA, typename GridB>
void expectSameLevelSet(const GridA& a, const GridB& b)
{
    EXPECT_EQ(a.tree().leafCount(), b.tree().leafCount());
    EXPECT_EQ(a.activeVoxelCount(), b.activeVoxelCount());
    EXPECT_EQ(voxelMismatches(a, b), Index64(0));
}

template<typename GridT>
void normalize(GridT& grid, math::BiasedGradientScheme spatial,
    math::TemporalIntegrationScheme temporal, bool useMask)
{
    using MaskTreeT = typename GridT::TreeType::template ValueConverter<ValueMask>::Type;
    tools::LevelSetTracker<GridT> tracker(grid);
    tracker.setSpatialScheme(spatial);
    tracker.setTemporalScheme(temporal);
    tracker.setNormCount(2);
    if (useMask) {
        MaskTreeT mask;
        for (auto voxel = grid.cbeginValueOn(); voxel; ++voxel) {
            if (voxel.getCoord().x() % 3 != 0) mask.setValueOn(voxel.getCoord());
        }
        tracker.template normalize<MaskTreeT>(&mask);
    } else {
        tracker.normalize();
    }
}

/// Returns the number of CFL steps taken.
template<typename GridT>
size_t advect(GridT& grid, math::BiasedGradientScheme spatial,
    math::TemporalIntegrationScheme temporal)
{
    using FieldT = tools::EnrightField<float>;
    FieldT field;
    tools::LevelSetAdvection<GridT, FieldT> advection(grid, field);
    advection.setSpatialScheme(spatial);
    advection.setTemporalScheme(temporal);
    advection.setTrackerSpatialScheme(spatial);
    return advection.advect(0.0f, 0.05f);
}

} // namespace

/// Tests the neighbor lookup of tools::lstrack::LeafStarStencil, in isolation.
///
/// The stencil replaces accessor lookups with direct reads from the current leaf and its six
/// face neighbors, so the risk is reading the wrong buffer or offset near a leaf boundary.
/// The math:: stencils it replaces serve as the reference.
///
/// Setup: grids of random values (see makeRandomGrid), with face neighbors that are leaves,
/// active tiles, inactive tiles and the background, for float and double, and for 8^3 leaves
/// and the smallest allowed leaves (4^3), where offsets of +-3 cross most of a leaf.
///
/// Asserts, for the 7, 13 and 19-point stencils, at every voxel and for both moveTo() forms:
/// - the center coordinate and every value equal those of the math:: stencil, bit for bit,
///   both by position, getValue(pos), and by offset, getValue<i, j, k>().
/// Also asserts, at compile time, which stencils the selector replaces and which it keeps.
TEST_F(TestStencils, testLeafStarStencilGather)
{
    using tools::lstrack::LeafStencil;
    using FloatGrid4 = Grid<tree::Tree4<float, 5, 4, 2>::Type>; // 4^3 leaves
    using FloatGrid2 = Grid<tree::Tree4<float, 5, 4, 1>::Type>; // 2^3 leaves

    // Leaves narrower than 4 voxels: offsets of +-3 would reach past the face neighbors.
    static_assert(std::is_same<
        LeafStencil<FloatGrid2, math::SevenPointStencil<FloatGrid2>>::Type,
        math::SevenPointStencil<FloatGrid2>>::value, "narrow leaves keep the math:: stencil");
    // Only star stencils can be replaced.
    static_assert(std::is_same<
        LeafStencil<FloatGrid, math::BoxStencil<FloatGrid>>::Type,
        math::BoxStencil<FloatGrid>>::value, "other stencils pass through");

    { SCOPED_TRACE("float, 8^3 leaves");  checkAllGathers<FloatGrid>(); }
    { SCOPED_TRACE("double, 8^3 leaves"); checkAllGathers<DoubleGrid>(); }
    { SCOPED_TRACE("float, 4^3 leaves");  checkAllGathers<FloatGrid4>(); }
}

/// Tests that LevelSetTracker and LevelSetAdvection give the same level sets with the leaf
/// stencil as with the math:: stencils. This catches what the test above cannot see, such as
/// a kernel that does not call setLeaf(), or that reads the wrong buffer between RK stages.
///
/// Setup: the same distorted sphere on a FloatGrid (leaf stencil) and on a ReferenceGrid
/// (math:: stencils), for all spatial schemes, so all three stencil sizes, and all temporal
/// schemes. Each is run through normalize() without and with a mask, which is a separate
/// code path, and through advect() in the Enright field.
///
/// Asserts, in each case:
/// - the two grids have the same leaf count and active voxel count, and in each voxel the same
///   active state and the same value within 1e-5 of a voxel (see voxelMismatches);
/// - the grid changed, so the comparison is not trivially true;
/// - advect() took the same number of CFL steps, and at least one.
TEST_F(TestStencils, testLeafStarStencilTrackerAndAdvection)
{
    static_assert(std::is_same<
        tools::lstrack::LeafStencil<FloatGrid, math::NineteenPointStencil<FloatGrid>>::Type,
        tools::lstrack::LeafStarStencil<FloatGrid, 3>>::value, "FloatGrid uses the leaf stencil");
    static_assert(std::is_same<
        tools::lstrack::LeafStencil<ReferenceGrid, math::NineteenPointStencil<ReferenceGrid>>::Type,
        math::NineteenPointStencil<ReferenceGrid>>::value, "ReferenceGrid uses the math:: stencil");

    const FloatGrid::Ptr input = makeDistortedSphere<FloatGrid>();

    for (const math::BiasedGradientScheme spatial : { math::FIRST_BIAS, math::SECOND_BIAS,
        math::THIRD_BIAS, math::WENO5_BIAS, math::HJWENO5_BIAS })
    {
        for (const math::TemporalIntegrationScheme temporal :
            { math::TVD_RK1, math::TVD_RK2, math::TVD_RK3 })
        {
            SCOPED_TRACE(math::biasedGradientSchemeToString(spatial) + ", "
                + math::temporalIntegrationSchemeToString(temporal));

            for (const bool useMask : { false, true }) {
                SCOPED_TRACE(useMask ? "normalize with mask" : "normalize");
                FloatGrid::Ptr star = makeDistortedSphere<FloatGrid>();
                ReferenceGrid::Ptr ref = makeDistortedSphere<ReferenceGrid>();
                normalize(*star, spatial, temporal, useMask);
                normalize(*ref, spatial, temporal, useMask);
                EXPECT_GT(voxelMismatches(*star, *input), Index64(0));
                expectSameLevelSet(*star, *ref);
            }

            SCOPED_TRACE("advect");
            FloatGrid::Ptr star = makeDistortedSphere<FloatGrid>();
            ReferenceGrid::Ptr ref = makeDistortedSphere<ReferenceGrid>();
            const size_t steps = advect(*star, spatial, temporal);
            EXPECT_EQ(steps, advect(*ref, spatial, temporal));
            EXPECT_GT(steps, size_t(0));
            EXPECT_GT(voxelMismatches(*star, *input), Index64(0));
            expectSameLevelSet(*star, *ref);
        }
    }
}
