using System.Numerics;
using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Learning;

public sealed record SpatialEncoderOptions(int Height = 64, int Width = 128, bool IncludeTerrain = true);

/// <summary>Exact public spatial metadata in observer coordinates. Set-aside and
/// undeployed locations are not exposed: PositionKnown is false and Pose is zero.
/// An already destroyed deployed unit may retain its public last-known pose while
/// OnBoard is false. Entity indices are stable native match indices.</summary>
public readonly record struct SpatialUnitPose(int Entity, int RelativeOwner, bool PositionKnown,
    bool OnBoard, bool IsSquadron, Pose Pose, Vector2 FootprintMillimetres, string ShapeId);

/// <summary>Public piece geometry for reconstruction; no opaque rule frame,
/// private command, future deck order or generic Payload store is exported.</summary>
public readonly record struct SpatialPiecePose(int Index, int Kind, int RelativeOwner, int SourceEntity,
    bool Active, Pose Pose, Vector2 SizeMillimetres, bool Flipped, string Id, string ShapeId);

/// <summary>Typed spatial model input. Encode returns owned arrays; EncodeBorrowed
/// returns reusable worker storage valid only until its next encode. Clone makes
/// an owned observation suitable for asynchronous inference or replay collation.
/// No per-squadron or per-ship raster planes are transported.</summary>
public sealed class SpatialEncoding
{
    public const string Schema = "alpharmada.spatial.v1";
    public const int ShipShipWidth = 152;
    public const int ShipSquadWidth = 56;
    public const int SquadSquadWidth = 12;
    public const int TerrainChannels = 4;
    public const int ShipCoordinateWidth = 4;
    public const int SquadronCoordinateWidth = 2;
    public const int SquadronRadiusWidth = 2;
    public int Observer { get; internal set; }
    public int ShipCount { get; }
    public int SquadronCount { get; }
    public int Height { get; }
    public int Width { get; }
    public Vector2 BoardSizeMillimetres { get; }
    public float[] ShipShipRelations { get; }
    public float[] ShipSquadRelations { get; }
    public float[] SquadSquadRelations { get; }
    public float[] ShipCoordinates { get; }
    public float[] SquadronCoordinates { get; }
    public float[] SquadronRadii { get; }
    public byte[] ShipOnBoard { get; }
    public byte[] SquadronOnBoard { get; }
    public byte[] TerrainRaster { get; }
    public SpatialUnitPose[] UnitPoses { get; }
    public SpatialPiecePose[] Pieces { get; internal set; } = [];
    public long TensorBytes => 4L * (ShipShipRelations.LongLength + ShipSquadRelations.LongLength + SquadSquadRelations.LongLength
        + ShipCoordinates.LongLength + SquadronCoordinates.LongLength + SquadronRadii.LongLength)
        + ShipOnBoard.LongLength + SquadronOnBoard.LongLength + TerrainRaster.LongLength;

    internal SpatialEncoding(int ships, int squadrons, Vector2 boardSize, SpatialEncoderOptions options)
    {
        ShipCount = ships; SquadronCount = squadrons; BoardSizeMillimetres = boardSize;
        Height = options.Height; Width = options.Width;
        ShipShipRelations = new float[checked(ships * ships * ShipShipWidth)];
        ShipSquadRelations = new float[checked(ships * squadrons * ShipSquadWidth)];
        SquadSquadRelations = new float[checked(squadrons * squadrons * SquadSquadWidth)];
        ShipCoordinates = new float[checked(ships * ShipCoordinateWidth)];
        SquadronCoordinates = new float[checked(squadrons * SquadronCoordinateWidth)];
        SquadronRadii = new float[checked(squadrons * SquadronRadiusWidth)];
        ShipOnBoard = new byte[ships]; SquadronOnBoard = new byte[squadrons];
        UnitPoses = new SpatialUnitPose[checked(ships + squadrons)];
        TerrainRaster = options.IncludeTerrain ? new byte[checked(TerrainChannels * Height * Width)] : [];
    }

    public SpatialEncoding Clone()
    {
        var copy = new SpatialEncoding(ShipCount, SquadronCount, BoardSizeMillimetres,
            new(Height, Width, TerrainRaster.Length != 0)) { Observer = Observer, Pieces = (SpatialPiecePose[])Pieces.Clone() };
        ShipShipRelations.CopyTo(copy.ShipShipRelations, 0); ShipSquadRelations.CopyTo(copy.ShipSquadRelations, 0);
        SquadSquadRelations.CopyTo(copy.SquadSquadRelations, 0); ShipCoordinates.CopyTo(copy.ShipCoordinates, 0);
        SquadronCoordinates.CopyTo(copy.SquadronCoordinates, 0); SquadronRadii.CopyTo(copy.SquadronRadii, 0);
        ShipOnBoard.CopyTo(copy.ShipOnBoard, 0); SquadronOnBoard.CopyTo(copy.SquadronOnBoard, 0);
        TerrainRaster.CopyTo(copy.TerrainRaster, 0); UnitPoses.CopyTo(copy.UnitPoses, 0);
        return copy;
    }
}
