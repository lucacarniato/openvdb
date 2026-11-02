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

/// Grid with leaves of width 2^Log2Dim and the standard tree depth.
template<typename ValueT, openvdb::Index Log2Dim>
using StarGrid = openvdb::Grid<typename openvdb::tree::Tree4<ValueT, 5, 4, Log2Dim>::Type>;

/// Same leaves as StarGrid, but different internal node sizes make it a distinct type, for
/// which the specializations below keep the tracker and advection on the math:: stencils.
template<typename ValueT, openvdb::Index Log2Dim>
using ReferenceGrid = openvdb::Grid<typename openvdb::tree::Tree4<ValueT, 4, 5, Log2Dim>::Type>;

} // namespace

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {
namespace lstrack {
template<> struct UseLeafStarStencil<ReferenceGrid<float, 3>>
{
    static constexpr bool value = false;
};
template<> struct UseLeafStarStencil<ReferenceGrid<float, 4>>
{
    static constexpr bool value = false;
};
template<> struct UseLeafStarStencil<ReferenceGrid<double, 3>>
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

/// Grid of random values, where leaves have every kind of face neighbor: another leaf,
/// the background, and active and inactive tiles. It spans negative and positive coordinates.
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
    const bool tileActive[6] = { true, false, false, true, true, false };
    for (int axis = 0, n = 0; axis < 3; ++axis) {
        for (int side = -1; side <= 1; side += 2, ++n) {
            Coord tile = node;
            tile[axis] += side * nodeDim;
            tree.addTile(/*level=*/1, tile, ValueT(11 + n), tileActive[n]);
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

    // Two blocks of leaves touching opposite corners of the node, with some holes in them.
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

/// Number of offsets within distance D on the axes where getValue<i, j, k>() differs.
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

/// Compare the leaf stencil to @a MathStencilT at every voxel of every leaf, active or not.
template<typename GridT, typename MathStencilT, int Radius>
void checkGather(const GridT& grid)
{
    using StarT = tools::lstrack::LeafStarStencil<GridT, Radius>;
    using LeafT = typename GridT::TreeType::LeafNodeType;
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
    Index64 voxels = 0, iterMismatches = 0, coordMismatches = 0;
    for (auto leaf = grid.tree().cbeginLeaf(); leaf; ++leaf) {
        star.setLeaf(*leaf);
        for (auto voxel = leaf->cbeginValueAll(); voxel; ++voxel, ++voxels) {
            math.moveTo(voxel);
            star.moveTo(voxel);
            iterMismatches += mismatches(star, math);
            star.moveTo(voxel.getCoord(), *voxel);
            coordMismatches += mismatches(star, math);
        }
    }
    EXPECT_EQ(voxels, grid.tree().leafCount() * LeafT::SIZE);
    EXPECT_GT(voxels, Index64(0));
    EXPECT_EQ(iterMismatches, Index64(0));
    EXPECT_EQ(coordMismatches, Index64(0));
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
    using ValueT = typename GridT::ValueType;
    typename GridT::Ptr grid = tools::createLevelSetSphere<GridT>(
        ValueT(0.15), Vec3f(0.35f), ValueT(1.0 / 64.0));
    for (auto voxel = grid->beginValueOn(); voxel; ++voxel) {
        const Coord c = voxel.getCoord();
        const double distortion =
            std::sin(0.7 * c.x()) * std::cos(0.9 * c.y()) * std::sin(1.1 * c.z());
        voxel.setValue(ValueT(double(*voxel) * (1.0 + 0.2 * distortion)));
    }
    return grid;
}

/// Number of voxels in the leaves of @a a whose state differs in @a b, or whose value differs
/// by more than 1e-5 of a voxel. The leaf stencil reads the same values (testLeafStarStencilGather
/// checks that bit for bit), but compilers can fuse multiply-adds differently in the two
/// instantiations, which changes the last bits of the results.
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
    EXPECT_EQ(voxelMismatches(b, a), Index64(0));
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

template<typename GridT>
size_t advect(GridT& grid, math::BiasedGradientScheme spatial,
    math::TemporalIntegrationScheme temporal)
{
    using FieldT = tools::EnrightField<typename GridT::ValueType>;
    FieldT field;
    tools::LevelSetAdvection<GridT, FieldT> advection(grid, field);
    advection.setSpatialScheme(spatial);
    advection.setTemporalScheme(temporal);
    advection.setTrackerSpatialScheme(spatial);
    return advection.advect(0.0, 0.05);
}

/// The leaf stencil (StarT) and the math:: stencils (RefT) must give the same level sets.
template<typename StarT, typename RefT>
void checkAgainstReference(math::BiasedGradientScheme spatial,
    math::TemporalIntegrationScheme temporal)
{
    const typename StarT::Ptr input = makeDistortedSphere<StarT>();
    {
        const typename RefT::Ptr ref = makeDistortedSphere<RefT>();
        expectSameLevelSet(*input, *ref);
    }
    for (const bool useMask : { false, true }) {
        SCOPED_TRACE(useMask ? "normalize with mask" : "normalize");
        typename StarT::Ptr star = makeDistortedSphere<StarT>();
        typename RefT::Ptr ref = makeDistortedSphere<RefT>();
        normalize(*star, spatial, temporal, useMask);
        normalize(*ref, spatial, temporal, useMask);
        EXPECT_GT(voxelMismatches(*star, *input), Index64(0)); // it did change the grid
        expectSameLevelSet(*star, *ref);
    }
    {
        SCOPED_TRACE("advect");
        typename StarT::Ptr star = makeDistortedSphere<StarT>();
        typename RefT::Ptr ref = makeDistortedSphere<RefT>();
        const size_t steps = advect(*star, spatial, temporal);
        EXPECT_EQ(steps, advect(*ref, spatial, temporal));
        EXPECT_GT(steps, size_t(0));
        EXPECT_GT(voxelMismatches(*star, *input), Index64(0));
        expectSameLevelSet(*star, *ref);
    }
}

} // namespace

TEST_F(TestStencils, testLeafStarStencilSelection)
{
    using namespace openvdb;
    using tools::lstrack::LeafStarStencil;
    using tools::lstrack::LeafStencil;
    using RefT = ReferenceGrid<float, 3>;
    using TinyLeafT = StarGrid<float, 1>; // leaves too narrow for offsets of +-3

    static_assert(std::is_same<LeafStencil<FloatGrid, math::NineteenPointStencil<FloatGrid>>::Type,
        LeafStarStencil<FloatGrid, 3>>::value, "");
    static_assert(std::is_same<LeafStencil<RefT, math::NineteenPointStencil<RefT>>::Type,
        math::NineteenPointStencil<RefT>>::value, "");
    static_assert(std::is_same<LeafStencil<TinyLeafT, math::SevenPointStencil<TinyLeafT>>::Type,
        math::SevenPointStencil<TinyLeafT>>::value, "");
    static_assert(std::is_same<LeafStencil<FloatGrid, math::BoxStencil<FloatGrid>>::Type,
        math::BoxStencil<FloatGrid>>::value, "");
    SUCCEED();
}

TEST_F(TestStencils, testLeafStarStencilGather)
{
    { SCOPED_TRACE("float, 8^3 leaves");  checkAllGathers<StarGrid<float, 3>>(); }
    { SCOPED_TRACE("double, 8^3 leaves"); checkAllGathers<StarGrid<double, 3>>(); }
    { SCOPED_TRACE("float, 4^3 leaves");  checkAllGathers<StarGrid<float, 2>>(); }
    { SCOPED_TRACE("float, 16^3 leaves"); checkAllGathers<StarGrid<float, 4>>(); }
}

TEST_F(TestStencils, testLeafStarStencilTrackerAndAdvection)
{
    using namespace openvdb;
    using StarT = StarGrid<float, 3>;
    using RefT = ReferenceGrid<float, 3>;

    for (const math::BiasedGradientScheme spatial : { math::FIRST_BIAS, math::SECOND_BIAS,
        math::THIRD_BIAS, math::WENO5_BIAS, math::HJWENO5_BIAS })
    {
        for (const math::TemporalIntegrationScheme temporal :
            { math::TVD_RK1, math::TVD_RK2, math::TVD_RK3 })
        {
            SCOPED_TRACE(math::biasedGradientSchemeToString(spatial) + ", "
                + math::temporalIntegrationSchemeToString(temporal));
            checkAgainstReference<StarT, RefT>(spatial, temporal);
        }
    }
    { SCOPED_TRACE("double");
      checkAgainstReference<StarGrid<double, 3>, ReferenceGrid<double, 3>>(
          math::HJWENO5_BIAS, math::TVD_RK2); }
    { SCOPED_TRACE("16^3 leaves");
      checkAgainstReference<StarGrid<float, 4>, ReferenceGrid<float, 4>>(
          math::HJWENO5_BIAS, math::TVD_RK2); }
}
