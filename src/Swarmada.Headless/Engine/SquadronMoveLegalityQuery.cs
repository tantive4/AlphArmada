using System.Numerics;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Engine;

/// <summary>Immutable scene-local ordinary squadron movement query. Callers must rebuild
/// after game state changes. Range/overlap arithmetic matches IsLegalMove, including
/// leaving the board; placement and card-specific movement use their own rules.</summary>
public sealed class SquadronMoveLegalityQuery
{
    private readonly Vector2 origin;
    private readonly float radius;
    private readonly float maximumDistance;
    private readonly bool restricted;
    private readonly PreparedRectangle[] ships;
    private readonly Circle[] squadrons;
    private readonly int shipCount;
    private readonly int squadronCount;
    public int EffectiveSpeed { get; }

    internal SquadronMoveLegalityQuery(GameContext game, int entity)
    {
        var source = game.SquadronBase(entity);
        origin = source.Center; radius = source.Radius;
        EffectiveSpeed = Math.Clamp(SquadronModule.EffectiveSpeed(game, entity), 0, 5);
        if (game.Flow.SquadronProgress is { MaximumMoveDistance: >= 0 } progress && progress.Entity == entity)
        { restricted = true; maximumDistance = progress.MaximumMoveDistance + 0.001f; }
        else maximumDistance = TabletopMeasurements.DistanceBand(EffectiveSpeed) + Geometry2D.DefaultTolerance;
        ships = new PreparedRectangle[game.State.Ships.Length];
        for (int ship = 0; ship < game.State.Ships.Length; ship++)
            if (!game.Topology.Ships[ship].IsStation && !game.IsDestroyed(ship) && !game.IsReserved(ship))
                ships[shipCount++] = new(game.ShipBase(ship));
        squadrons = new Circle[game.State.Squadrons.Length];
        for (int index = 0; index < game.State.Squadrons.Length; index++)
        {
            int other = game.State.Ships.Length + index;
            if (other != entity && !game.IsDestroyed(other) && !game.IsReserved(other))
                squadrons[squadronCount++] = game.SquadronBase(other);
        }
    }

    public bool IsLegal(Vector2 destination)
    {
        // The unrestricted native ruler rejects nonfinite coordinates with an
        // argument error; the restricted path rejects them via its distance test.
        if (!restricted) Geometry2D.RequireFinite(destination, nameof(destination));
        if (!(Vector2.Distance(origin, destination) <= maximumDistance)) return false;
        if (!float.IsFinite(destination.X) || !float.IsFinite(destination.Y)) return false;
        for (int i = 0; i < shipCount; i++)
            if (!(radius - Vector2.Distance(destination, ships[i].ClosestPoint(destination)) <= Geometry2D.DefaultTolerance)) return false;
        for (int i = 0; i < squadronCount; i++)
            if (!(radius + squadrons[i].Radius - Vector2.Distance(destination, squadrons[i].Center) <= Geometry2D.DefaultTolerance)) return false;
        return true;
    }

    // Reuse the same float transforms as OrientedRectangle.ClosestPoint while
    // computing each blocker's trigonometry only once for the whole domain.
    private readonly struct PreparedRectangle
    {
        private readonly Vector2 center;
        private readonly Vector2 halfSize;
        private readonly float cosine;
        private readonly float sine;
        public PreparedRectangle(OrientedRectangle rectangle)
        {
            center = rectangle.Center; halfSize = rectangle.HalfSize;
            float radians = rectangle.Pose.YawDegrees * (MathF.PI / 180);
            cosine = MathF.Cos(radians); sine = MathF.Sin(radians);
        }
        public Vector2 ClosestPoint(Vector2 point)
        {
            var offset = point - center;
            var local = new Vector2(cosine * offset.X - sine * offset.Y, sine * offset.X + cosine * offset.Y);
            var clamped = Vector2.Clamp(local, -halfSize, halfSize);
            return center + new Vector2(cosine * clamped.X + sine * clamped.Y, -sine * clamped.X + cosine * clamped.Y);
        }
    }
}
