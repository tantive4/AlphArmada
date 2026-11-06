using Swarmada.Headless.Geometry;

namespace Swarmada.Headless.Engine;

public sealed record MercyMissionAward(int SourceShip, int AssignedShip, int Player, int Points, string Reason);

public static partial class MapAbilityEffects
{
    /// <summary>Evaluates Mercy Mission before final scores and winner selection. The award
    /// is added by TournamentRules without incrementing the persistent objective-point pool,
    /// so snapshots and repeated score queries cannot award it twice.</summary>
    public static IReadOnlyList<MercyMissionAward> GetMercyMissionAwards(GameContext game)
    {
        var awards = new List<MercyMissionAward>();
        foreach (var source in game.Topology.Ships)
            foreach (var card in game.UpgradesFor(source.EntityIndex).Where(c => c.Active && c.Definition.id == "mercy-mission"))
                foreach (string runtimeId in card.StoredUnitIds)
                {
                    if (!game.Topology.EntityByRuntimeId.TryGetValue(runtimeId, out int assigned) || !game.IsShip(assigned)) continue;
                    int owner = game.State.Owners[source.EntityIndex];
                    if (TournamentRules.CountsAsDestroyed(game, assigned))
                    { awards.Add(new(source.EntityIndex, assigned, 3 - owner, 15, "assigned-ship-destroyed")); continue; }
                    var zone = game.Topology.DeploymentZones.Single(z => z.Owner == 3 - owner);
                    var bounds = new OrientedRectangle(new((zone.MinimumMillimetres + zone.MaximumMillimetres) / 2), zone.MaximumMillimetres - zone.MinimumMillimetres);
                    if (PlayAreaGeometry.Contains(bounds, game.ShipBase(assigned))) awards.Add(new(source.EntityIndex, assigned, owner, 40, "assigned-ship-in-enemy-deployment-zone"));
                }
        return awards;
    }
}
