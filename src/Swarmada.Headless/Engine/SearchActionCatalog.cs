using System.Globalization;
using System.Numerics;
using System.Text;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Geometry;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

/// <summary>A caller-owned finite gameplay domain, not an engine movement policy.</summary>
public sealed record ActionCatalogProfile(int SquadronGridWidth = 33, int ContactDirections = 128,
    int PlacementColumns = 33, int PlacementRows = 17, int PlacementHeadings = 16,
    bool IncludeUiNavigation = false)
{
    public const string Version = "search-finite-catalog-v2";
}

public sealed class SearchCoverageException(string message) : InvalidOperationException(message);

/// <summary>Expands every offer over a finite domain and masks exact candidates.
/// Does not rank candidates or substitute a fallback. One instance per worker.</summary>
public sealed class ActionCatalog
{
    private readonly ActionCatalogProfile profile;
    private readonly DiscreteActionAdapter validator = new();
    public long CandidatesValidated { get; private set; }
    public long CatalogsBuilt { get; private set; }
    public ActionCatalog(ActionCatalogProfile? profile = null)
    {
        this.profile = profile ?? new();
        if (this.profile.SquadronGridWidth < 3 || this.profile.SquadronGridWidth % 2 == 0 ||
            this.profile.ContactDirections < 8 || this.profile.PlacementColumns < 2 ||
            this.profile.PlacementRows < 2 || this.profile.PlacementHeadings < 1)
            throw new ArgumentOutOfRangeException(nameof(profile));
    }

    public IReadOnlyList<GameAction> Build(GameSession game, GameDecision decision)
    {
        if (decision.Kind != DecisionKind.Player) return [];
        var complete = new List<GameAction>(); var spatial = new List<DiscreteActionCandidate>();
        var seen = new HashSet<GameAction>(ExactGameActionComparer.Instance);
        foreach (var option in decision.Actions)
        {
            // Deselect is the recovered UI's back button for an uncommitted unit
            // selection. MCTS already explores alternative units at their parent;
            // retaining this backedge permits unbounded select/deselect loops.
            // Real native effect cancellation choices are left untouched.
            if (!profile.IncludeUiNavigation && option.Template.AbilityId == "squadron.deselect") continue;
            bool control = profile.IncludeUiNavigation && decision.Phase == GamePhase.SquadronActivation && option.Template.Kind == GameActionKind.MoveSquadron;
            if ((!option.AcceptsPosition || control) && !option.AcceptsCourse && !option.AcceptsYaw && !option.AcceptsFlip)
            {
                if (seen.Add(option.Template)) complete.Add(option.Template.DeepClone());
                continue;
            }
            bool ordinarySquadronMove = OrdinarySquadronMoveOffer(game, decision, option);
            var moveQuery = ordinarySquadronMove ? SquadronModule.PrepareMoveLegality(game.Context, option.Template.Source) : null;
            foreach (var action in Expand(game, decision, option))
            {
                if (!seen.Add(action)) continue;
                if (ordinarySquadronMove)
                {
                    CandidatesValidated++;
                    // Native GetDecision established CanMoveNow. The ordinary
                    // Apply branch tests this exact predicate before opening any
                    // reaction window or committing movement. Never apply this
                    // shortcut to a card-specific move/placement/displacement.
                    if (moveQuery!.IsLegal(action.Position!.Value)) complete.Add(action);
                }
                else spatial.Add(new(spatial.Count, action));
            }
        }
        CatalogsBuilt++;
        if (spatial.Count > 0)
        {
            var mask = validator.GetDecision(game, spatial); CandidatesValidated += spatial.Count;
            if (mask.Kind == DecisionKind.InvalidSetup)
                throw new SearchCoverageException("Native setup invalid; discard/retry outside search: " + mask.InvalidSetup);
            complete.AddRange(mask.Actions.Where(status => status.Legal).Select(status => status.Action));
        }
        if (complete.Count == 0)
            throw new SearchCoverageException($"Empty finite catalog at {decision.Phase}: " +
                (game.Context.Flow.Continuations.LastOrDefault()?.ToString() ?? "no continuation"));
        return complete;
    }

    private static bool OrdinarySquadronMoveOffer(GameSession game, GameDecision decision, GameActionOption option)
    {
        var action = option.Template; var context = game.Context;
        if (decision.Phase is not (GamePhase.SquadronActivation or GamePhase.SquadronMove) ||
            action.Kind != GameActionKind.MoveSquadron || action.AbilityId.Length != 0 ||
            !option.AcceptsPosition || option.AcceptsYaw || option.AcceptsCourse || option.AcceptsFlip ||
            context.Flow.SquadronProgress is not { Stage: 1 } progress || progress.Entity != action.Source ||
            action.Source != context.Flow.ActiveSquadron || action.Target != -1 || context.IsDestroyed(action.Source)) return false;
        // SquadronMove's native destination window is a UI continuation and offers
        // its template unconditionally. Recheck CanMoveNow there as well; a stale
        // or card-altered context must take the full native validation path.
        var state = context.State.Squadrons[context.SquadronIndex(action.Source)];
        return context.Flow.SquadronCanMove && !progress.Moved && (progress.CanMoveAndAttack || !progress.Attacked)
            && !state.TemporaryAttackOnly && !SquadronModule.IsPreventedFromMoving(context, action.Source)
            && (!progress.Commanded || context.IsShip(context.Flow.ActiveShip));
    }

    private IEnumerable<GameAction> Expand(GameSession game, GameDecision decision, GameActionOption option)
    {
        var action = option.Template;
        if (option.AcceptsCourse)
        {
            if (decision.CourseConstraints is not { } constraints)
                throw new SearchCoverageException("Course offer has no analytic constraints.");
            if (constraints.JointAllowances.Count > 6) throw new SearchCoverageException("Course length exceeds finite catalog limit six.");
            foreach (var pattern in MovementModule.GetChartValidCourses(constraints))
            {
                var course = pattern.ToArray();
                foreach (var side in course.All(click => click == 0) ? new[] { GuideSide.Center } : new[] { GuideSide.Port, GuideSide.Starboard })
                    yield return action with { Course = course, Speed = constraints.DialedSpeed, Index = (int)side };
            }
            yield break;
        }
        if (option.AcceptsFlip)
            throw new SearchCoverageException("Obstacle/creature placement needs its registered contact domain: " + action.AbilityId);
        if (action.Kind == GameActionKind.DisplaceSquadron)
        {
            var ship = game.Context.ShipBase(action.Target); var squad = game.Context.SquadronBase(action.Source);
            for (int i = 0; i < profile.ContactDirections; i++)
            {
                float angle = 2 * MathF.PI * i / profile.ContactDirections;
                var desired = ship.Pose.Position + new Vector2(MathF.Cos(angle), MathF.Sin(angle)) * 1000;
                yield return action with { Position = MovementModule.TouchingPlacement(ship, squad, desired) };
            }
            foreach (var other in game.Context.Topology.Squadrons)
            {
                if (other.EntityIndex == action.Source || game.Context.Flow.PendingDisplacements.Contains(other.EntityIndex)) continue;
                var anchor = game.Context.SquadronBase(other.EntityIndex);
                for (int i = 0; i < profile.ContactDirections; i++)
                {
                    float angle = 2 * MathF.PI * i / profile.ContactDirections;
                    yield return action with { Position = anchor.Center + new Vector2(MathF.Cos(angle), MathF.Sin(angle)) * (anchor.Radius + squad.Radius) };
                }
            }
            yield break;
        }
        if (action.Kind == GameActionKind.MoveSquadron && !option.AcceptsYaw)
        {
            int entity = game.Context.IsShip(action.Source) ? action.Target : action.Source;
            if (entity < game.Context.State.Ships.Length || entity >= game.Context.Topology.EntityCount)
                throw new SearchCoverageException("Cannot identify moving squadron for " + action.AbilityId);
            float pitch = 2 * TabletopMeasurements.DistanceBand(5) / profile.SquadronGridWidth;
            float radius = TabletopMeasurements.DistanceBand(SquadronModule.EffectiveSpeed(game.Context, entity));
            int half = profile.SquadronGridWidth / 2;
            for (int x = -half; x <= half; x++) for (int y = -half; y <= half; y++)
            {
                var offset = new Vector2(x * pitch, y * pitch);
                if (offset.LengthSquared() <= radius * radius + .0001f)
                    yield return action with { Position = game.Context.State.Poses[entity].Position + offset };
            }
            yield break;
        }
        if (!option.AcceptsPosition) throw new SearchCoverageException("Unregistered yaw domain: " + action.AbilityId);
        for (int x = 0; x < profile.PlacementColumns; x++) for (int y = 0; y < profile.PlacementRows; y++)
        {
            var point = new Vector2(TabletopMeasurements.StandardBoardWidth * (x / (float)(profile.PlacementColumns - 1) - .5f),
                TabletopMeasurements.StandardBoardLength * (y / (float)(profile.PlacementRows - 1) - .5f));
            for (int h = 0; h < (option.AcceptsYaw ? profile.PlacementHeadings : 1); h++)
                yield return action with { Position = point, YawDegrees = option.AcceptsYaw ? h * 360f / profile.PlacementHeadings : action.YawDegrees };
        }
    }

    /// <summary>Persistent semantic identity; no revision/session token or hash collision.</summary>
    public static string Key(GameAction action)
    {
        static string F(float value) => value.ToString("R", CultureInfo.InvariantCulture);
        var text = new StringBuilder();
        text.Append((int)action.Kind).Append('|').Append(action.Source).Append('|').Append(action.Target)
            .Append('|').Append((int)action.FromZone).Append('|').Append((int)action.ToZone).Append('|').Append((int)action.Command)
            .Append('|').Append((int)action.SpendMode).Append('|').Append((int)action.Token).Append('|').Append(action.Index)
            .Append('|').Append(action.Amount).Append('|').Append(action.Speed).Append('|').Append(F(action.YawDegrees))
            .Append('|').Append(action.Flipped ? 1 : 0).Append('|').Append(Convert.ToBase64String(Encoding.UTF8.GetBytes(action.AbilityId)));
        text.Append('|').Append(action.Position is { } p ? F(p.X) + "," + F(p.Y) : "_");
        text.Append('|').Append(action.Course is { } c ? string.Join(',', c) : "_");
        text.Append('|').Append(action.DiceIndices is { } d ? string.Join(',', d) : "_");
        text.Append('|').Append(action.Dice is { } dice ? string.Join(',', dice.Select(die =>
            ((int)die.Color).ToString(CultureInfo.InvariantCulture) + ":" + ((int)die.Face).ToString(CultureInfo.InvariantCulture))) : "_");
        return text.ToString();
    }
}
