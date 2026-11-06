#nullable disable
using System;
using System.Collections.Generic;
using Armada.Core;
using Armada.Data;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class RemainingUpgradeRules
{
	public const string AdmiralKonstantine = "admiral-konstantine-commander";

	public const string AdmiralRaddus = "admiral-raddus-commander";

	public const string AdmiralSloane = "admiral-sloane-commander";

	public const string AdmiralTarkin = "admiral-tarkin-commander";

	public const string AdmiralTrench = "admiral-trench-commander";

	public const string AnakinSkywalker = "anakin-skywalker-commander";

	public const string BailOrgana = "bail-organa-commander";

	public const string CommanderSato = "commander-sato-commander";

	public const string CountDooku = "count-dooku-commander";

	public const string EmperorPalpatineCommander = "emperor-palpatine-commander";

	public const string GeneralDraven = "general-draven-commander";

	public const string GeneralGrievous = "general-grievous-commander";

	public const string GeneralRieekan = "general-rieekan-commander";

	public const string GrandAdmiralThrawn = "grand-admiral-thrawn-commander";

	public const string LuminaraUnduli = "luminara-unduli-commander";

	public const string MoffJerjerrod = "moff-jerjerrod-commander";

	public const string G8ExperimentalProjector = "g-8-experimental-projector";

	public const string G7XGravWellProjector = "g7-x-grav-well-projector";

	public const string GravShiftReroute = "grav-shift-reroute";

	public const string MagniteCrystalTractorBeamArray = "magnite-crystal-tractor-beam-array";

	public const string MercyMission = "mercy-mission";

	public const string FlagBridge = "flag-bridge";

	public const string HyperspaceRings = "hyperspace-rings";

	public const string HyperwaveSignalBoost = "hyperwave-signal-boost";

	public const string ProximityMines = "proximity-mines";

	public const string RapidLaunchBays = "rapid-launch-bays";

	public const string SphaT = "spha-t";

	public const string AdarTallon = "adar-tallon-officer";

	public const string AdmiralChiraneau = "admiral-chiraneau-officer";

	public const string AdmiralOzzelOfficer = "admiral-ozzel-officer";

	public const string AdmiralTitus = "admiral-titus-officer";

	public const string AhsokaTanoRebel = "ahsoka-tano-officer-rebel";

	public const string AhsokaTanoRepublic = "ahsoka-tano-officer-republic";

	public const string BarrissOffee = "barriss-offee-officer";

	public const string ChartOfficer = "chart-officer";

	public const string CloneCaptainSilver = "clone-captain-silver-officer";

	public const string CloneCommanderWolffe = "clone-commander-wolffe-officer";

	public const string EmperorPalpatineOfficer = "emperor-palpatine-officer";

	public const string EzraBridger = "ezra-bridger-officer";

	public const string FlightCommander = "flight-commander-officer";

	public const string PasselArgente = "passel-argente-officer";

	public const string RuneHaako = "rune-haako-officer";

	public const string SabineWren = "sabine-wren-officer";

	public const string SanHill = "san-hill-officer";

	public const string GrandInquisitor = "the-grand-inquisitor-officer";

	public const string Tikkes = "tikkes-officer";

	public const string WedgeAntilles = "wedge-antilles-officer";

	public const string FlechetteTorpedoes = "flechette-torpedoes";

	public const string OrdnancePods = "ordnance-pods";

	public const string EngineTechs = "engine-techs";

	public const string FighterCoordinationTeam = "fighter-coordination-team";

	public const string NavTeam = "nav-team";

	public const string CloneGunners = "clone-gunners";

	public const string GunneryTeam = "gunnery-team";

	public const string RuthlessStrategists = "ruthless-strategists";

	public const string BoardingEngineers = "boarding-engineers-boarding-team";

	public const string BoardingTroopers = "boarding-troopers-boarding-team";

	public const string ChamSyndulla = "cham-syndulla-boarding-team";

	public const string DarthVaderBoarding = "darth-vader-boarding-team";

	public const string ShrivSurgaav = "shriv-surgaav-boarding-team";

	public const string Aspiration = "aspiration-title";

	public const string BeastOfBurden = "beast-of-burden-title";

	public const string Cataclysm = "cataclysm-title";

	public const string Centicore = "centicore-title";

	public const string Chimaera = "chimaera-title";

	public const string Corvus = "corvus-title";

	public const string Demolisher = "demolisher-title";

	public const string GildedAegis = "gilded-aegis-title";

	public const string HandOfJustice = "hand-of-justice-title";

	public const string Harrow = "harrow-title";

	public const string Impetuous = "impetuous-title";

	public const string Independence = "independence-title";

	public const string Interdictor = "interdictor-title";

	public const string InvisibleHand = "invisible-hand-title";

	public const string Liberator = "liberator-title";

	public const string NevootaBee = "nevoota-bee-title";

	public const string Profundity = "profundity-title";

	public const string QuantumStorm = "quantum-storm-title";

	public const string Rakehell = "rakehell-title";

	public const string Ravager = "ravager-title";

	public const string Resolute = "resolute-title";

	public const string Sovereign = "sovereign-title";

	public const string Squall = "squall-title";

	public const string Suppressor = "suppressor-title";

	public const string SwiftReturn = "swift-return-title";

	public const string TantiveIv = "tantive-iv-title";

	public const string Triumphant = "triumphant-title";

	public const string Yavaris = "yavaris-title";

	public const string StartActivationPrefix = "remaining.start-activation.";

	public const string CommandRevealPrefix = "remaining.command-reveal.";

	public const string AttackDicePrefix = "remaining.attack-dice.";

	public const string DefensePrefix = "remaining.defense.";

	public const string DetermineCoursePrefix = "remaining.determine-course.";

	public const string AfterManeuverPrefix = "remaining.after-maneuver.";

	public const string EndActivationPrefix = "remaining.end-activation.";

	public const string ShipPhasePrefix = "remaining.ship-phase.";

	public const string StatusPrefix = "remaining.status.";

	public const string SquadronActivationPrefix = "remaining.squadron-activation.";

	public const string SquadronDicePrefix = "remaining.squadron-dice.";

	private static Dictionary<int, ShipCommand> TrenchCommands => GameScope.Current.Flow.SourceRules.TrenchCommands;

	private static Dictionary<int, ShipCommand> DravenCommands => GameScope.Current.Flow.SourceRules.DravenCommands;

	private static Dictionary<int, ShipCommand> ThrawnCommands => GameScope.Current.Flow.SourceRules.ThrawnCommands;

	private static Dictionary<int, ShipCommand> AdmiralTarkinDials => GameScope.Current.Flow.SourceRules.AdmiralTarkinDials;

	private static Dictionary<int, DefenseTokenType> PalpatineTokens => GameScope.Current.Flow.SourceRules.PalpatineTokens;

	private static Dictionary<int, HashSet<DefenseTokenType>> PalpatineRemaining => GameScope.Current.Flow.SourceRules.PalpatineRemaining;

	public static void ResetRoundEffects()
	{
		TrenchCommands.Clear();
		DravenCommands.Clear();
		ThrawnCommands.Clear();
		AdmiralTarkinDials.Clear();
		PalpatineTokens.Clear();
	}

	public static void ActivateTrench(int ownerId, ShipCommand command)
	{
		TrenchCommands[ownerId] = command;
	}

	public static void ActivateDraven(int ownerId, ShipCommand command)
	{
		DravenCommands[ownerId] = command;
	}

	public static void ActivateThrawn(int ownerId, ShipCommand command)
	{
		ThrawnCommands[ownerId] = command;
	}

	public static void ActivateAdmiralTarkinDial(int ownerId, ShipCommand command)
	{
		AdmiralTarkinDials[ownerId] = command;
	}

	public static bool TryGetAdmiralTarkinDial(int ownerId, out ShipCommand command)
	{
		return AdmiralTarkinDials.TryGetValue(ownerId, out command);
	}

	public static bool TryConsumeAdmiralTarkinDial(int ownerId, out ShipCommand command)
	{
		if (!AdmiralTarkinDials.TryGetValue(ownerId, out command))
		{
			return false;
		}
		AdmiralTarkinDials.Remove(ownerId);
		return true;
	}

	public static void ActivatePalpatine(int ownerId, DefenseTokenType token)
	{
		PalpatineTokens[ownerId] = token;
	}

	public static bool TryConsumePalpatineToken(int ownerId, out DefenseTokenType token)
	{
		foreach (DefenseTokenType remainingPalpatineToken in GetRemainingPalpatineTokens(ownerId))
		{
			if (TryActivatePalpatineToken(ownerId, remainingPalpatineToken))
			{
				token = remainingPalpatineToken;
				return true;
			}
		}
		token = DefenseTokenType.Brace;
		return false;
	}

	public static IReadOnlyCollection<DefenseTokenType> GetRemainingPalpatineTokens(int ownerId)
	{
		if (!PalpatineRemaining.TryGetValue(ownerId, out var value))
		{
			value = new HashSet<DefenseTokenType>
			{
				DefenseTokenType.Brace,
				DefenseTokenType.Redirect,
				DefenseTokenType.Evade,
				DefenseTokenType.Contain,
				DefenseTokenType.Scatter,
				DefenseTokenType.Salvo
			};
			PalpatineRemaining[ownerId] = value;
		}
		return value;
	}

	public static bool TryActivatePalpatineToken(int ownerId, DefenseTokenType token)
	{
		GetRemainingPalpatineTokens(ownerId);
		if (!PalpatineRemaining[ownerId].Remove(token))
		{
			return false;
		}
		ActivatePalpatine(ownerId, token);
		return true;
	}

	public static bool TrenchUpgradesToken(int ownerId, ShipCommand command)
	{
		if (TrenchCommands.TryGetValue(ownerId, out var value))
		{
			return value == command;
		}
		return false;
	}

	public static bool DravenDowngradesDial(int ownerId, ShipCommand command)
	{
		int key = ((ownerId != 1) ? 1 : 2);
		if (DravenCommands.TryGetValue(key, out var value))
		{
			return value == command;
		}
		return false;
	}

	public static bool TryGetThrawnCommand(int ownerId, out ShipCommand command)
	{
		return ThrawnCommands.TryGetValue(ownerId, out command);
	}

	public static bool PalpatinePunishesToken(int defenderOwnerId, DefenseTokenType token)
	{
		int key = ((defenderOwnerId != 1) ? 1 : 2);
		if (PalpatineTokens.TryGetValue(key, out var value))
		{
			return value == token;
		}
		return false;
	}

	public static bool FleetHas(ShipView ship, string cardId)
	{
		return CommanderRules.FleetHasCommander(ship, cardId);
	}

	public static UpgradeCardRuntimeState FindFleetCard(ShipView ship, string cardId)
	{
		return CommanderRules.FindFleetUpgrade(ship, cardId);
	}

	public static ShipView FindCardCarrier(int ownerId, string cardId)
	{
		foreach (ShipView item in FindShips())
		{
			if (item != null && !item.IsDestroyed && item.OwnerId == ownerId)
			{
				ShipLoadoutState loadout = item.Loadout;
				if (loadout != null && loadout.HasActive(cardId))
				{
					return item;
				}
			}
		}
		return null;
	}

	public static IEnumerable<ShipView> FindShips()
	{
		return GameScope.Current.SceneShipViews;
	}

	public static IEnumerable<SquadronView> FindSquadrons()
	{
		return GameScope.Current.SceneSquadronViews;
	}

	public static bool IsFlotilla(ShipView ship)
	{
		return ship is not null && ship.Game.Topology.Ships[ship.Index].CatalogEntry.Metadata.IsFlotilla == true;
	}

	public static int SizeRank(ShipView ship)
	{
		return (ship?.SizeClass ?? string.Empty).Trim().ToLowerInvariant() switch
		{
			"small" => 1,
			"medium" => 2,
			"large" => 3,
			"huge" => 4,
			_ => 0,
		};
	}

	public static ShipView NearestEnemy(ShipView source, float maximumDistance)
	{
		ShipView result = null;
		float num = float.MaxValue;
		foreach (ShipView item in FindShips())
		{
			if (!(item == null) && !item.IsDestroyed && !(source == null) && item.OwnerId != source.OwnerId)
			{
				float num2 = RaidRules.FootprintDistance(source, item);
				if (!(num2 > maximumDistance) && !(num2 >= num))
				{
					num = num2;
					result = item;
				}
			}
		}
		return result;
	}

	public static ShipView NearestFriendly(ShipView source, float maximumDistance, bool includeSource = false)
	{
		ShipView result = null;
		float num = float.MaxValue;
		foreach (ShipView item in FindShips())
		{
			if (!(item == null) && !item.IsDestroyed && !(source == null) && item.OwnerId == source.OwnerId && (includeSource || !(item == source)))
			{
				float num2 = RaidRules.FootprintDistance(source, item);
				if (!(num2 > maximumDistance) && !(num2 >= num))
				{
					num = num2;
					result = item;
				}
			}
		}
		return result;
	}

	public static bool HasFriendlySquadronNear(ShipView defender, int ownerId, float maximumDistance)
	{
		foreach (SquadronView item in FindSquadrons())
		{
			if (!(item == null) && !item.IsDestroyed && item.OwnerId == ownerId && SquadronAbilityEngine.FootprintDistance(item, defender) <= maximumDistance)
			{
				return true;
			}
		}
		return false;
	}

	public static DefenseTokenType? FirstReadyDefenseToken(ShipView ship)
	{
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			if ((object)ship != null && ship.State?.IsDefenseTokenReady(value) == true)
			{
				return value;
			}
		}
		return null;
	}

	public static bool HasEffectiveIgnition(ShipView ship)
	{
		return EffectiveIgnitionProfiles(ship).Count > 0;
	}

	public static IReadOnlyList<IgnitionProfile> EffectiveIgnitionProfiles(ShipView ship)
	{
		if (ship?.Definition == null)
		{
			return Array.Empty<IgnitionProfile>();
		}
		ShipLoadoutState loadout = ship.Loadout;
		if (loadout != null && loadout.HasActive("spha-t"))
		{
			return new IgnitionProfile[3]
			{
				CreateSphaTProfile(ship, HullZone.Front),
				CreateSphaTProfile(ship, HullZone.Left),
				CreateSphaTProfile(ship, HullZone.Right)
			};
		}
		IgnitionProfile ignition = ship.Definition.Ignition;
		if (ignition == null || !ignition.IsConfigured)
		{
			return Array.Empty<IgnitionProfile>();
		}
		return new IgnitionProfile[1] { ignition };
	}

	public static bool TryGetEffectiveIgnitionProfile(ShipView ship, HullZone hullZone, out IgnitionProfile profile)
	{
		foreach (IgnitionProfile item in EffectiveIgnitionProfiles(ship))
		{
			if (item.AssociatedHullZone == hullZone)
			{
				profile = item;
				return true;
			}
		}
		profile = null;
		return false;
	}

	public static bool TryResolveEffectiveIgnitionProfile(ShipView ship, Vector3 worldPosition, out IgnitionProfile profile)
	{
		profile = null;
		float num = float.MaxValue;
		foreach (IgnitionProfile item in EffectiveIgnitionProfiles(ship))
		{
			if (IgnitionAttackGeometry.ContainsPointInSpecialArc(ship.transform.position, ship.transform.rotation, item, worldPosition))
			{
				IgnitionAttackGeometry.GetWorldSpecialArc(ship.transform.rotation, item, out var _, out var _, out var forward, out var _);
				Vector3 vector = EngineVectors.ProjectOnPlane(worldPosition - ship.transform.position, Vector3.UnitY);
				float num2 = ((vector.LengthSquared() < 0.0001f) ? float.MaxValue : EngineVectors.Angle(forward, vector.NormalizeSafe()));
				if (!(num2 >= num))
				{
					num = num2;
					profile = item;
				}
			}
		}
		return profile != null;
	}

	public static bool CanPlaceEffectiveIgnitionTarget(ShipView ship)
	{
		if (!HasEffectiveIgnition(ship))
		{
			return false;
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState = ship.Loadout?.Find("spha-t");
		if (upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active)
		{
			return !upgradeCardRuntimeState.Exhausted;
		}
		return true;
	}

	public static bool CommitEffectiveIgnitionTargetPlacement(ShipView ship, HullZone hullZone)
	{
		if (!TryGetEffectiveIgnitionProfile(ship, hullZone, out var _))
		{
			return false;
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState = ship.Loadout?.Find("spha-t");
		if (upgradeCardRuntimeState == null || !upgradeCardRuntimeState.Active)
		{
			return true;
		}
		return upgradeCardRuntimeState.Exhaust();
	}

	private static IgnitionProfile CreateSphaTProfile(ShipView ship, HullZone hullZone)
	{
		ship.GetHullZoneBoundaryDirections(hullZone, out var start, out var end);
		Vector3 vector = ship.transform.InverseTransformDirection(start);
		Vector3 vector2 = ship.transform.InverseTransformDirection(end);
		return new IgnitionProfile(hullZone, new BatteryArmament(0, 5, 1), new Vector2(vector.X, vector.Z).NormalizeSafe(), new Vector2(vector2.X, vector2.Z).NormalizeSafe(), new RangeBand[1], "onager-ignition-targeting");
	}

	public static bool IsBoardingTeam(string cardId)
	{
		switch (cardId)
		{
		default:
			return cardId == "shriv-surgaav-boarding-team";
		case "boarding-engineers-boarding-team":
		case "boarding-troopers-boarding-team":
		case "cham-syndulla-boarding-team":
		case "darth-vader-boarding-team":
			return true;
		}
	}

	public static bool IsIgnitionUpgrade(string cardId)
	{
		if (!(cardId == "spha-t") && !(cardId == "cataclysm-title"))
		{
			return cardId == "magnite-crystal-tractor-beam-array";
		}
		return true;
	}

	public static bool IsFleetCommandIconGrant(string cardId)
	{
		if (!(cardId == "flag-bridge") && !(cardId == "chimaera-title"))
		{
			return cardId == "liberator-title";
		}
		return true;
	}
}
