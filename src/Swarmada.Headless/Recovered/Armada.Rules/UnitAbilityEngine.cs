#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;

using Armada.Core;
using Swarmada.Headless.Engine;
using System.Numerics;
using Mathf = Swarmada.Headless.Engine.EngineMath;

namespace Armada.Rules;

public static class UnitAbilityEngine
{
	public const string ExternalRacksAction = "upgrade.external-racks.add-black-dice";

	public const string B2RocketTroopersRaidAction = "upgrade.b2-rocket-troopers.place-raid";

	public const string IdenVersioRaidAction = "upgrade.iden-versio.place-raid";

	public const string JynErsoRaidDialAction = "upgrade.jyn-erso.place-raid-with-dial";

	public const string JynErsoRaidTokenAction = "upgrade.jyn-erso.place-raid-with-token";

	public const string CommanderBeckActionPrefix = "upgrade.commander-beck.resolve.";

	public const string CommandChangeActionPrefix = "command.change.";

	public const string VeteranCaptainActionPrefix = "command.veteran-captain.gain.";

	public const string LiaisonActionPrefix = "command.liaison.";

	public const string DiscardTopCommandActionPrefix = "command.discard-top.";

	public const string WalexBlissexActionPrefix = "defense.walex-blissex.recover.";

	public const string RaymusAntillesAction = "command.raymus-antilles.gain-matching";

	public const string TaskmasterGrintAction = "command.taskmaster-grint.gain-matching";

	public const string TokenAsDialActionPrefix = "command.token-as-dial.";

	public const string SupportOfficerAction = "command.support-officer.discard-stack";

	public const string PursuantAction = "command.pursuant.resolve-squadron";

	public const string AsajjVentress = "asajj-ventress-officer";

	public const string HotLanding = "hot-landing";

	public const string LockContainAction = "attack.accuracy.lock-contain";

	public const string LockScatterAction = "attack.accuracy.lock-ship-scatter";

	public const string LockSalvoAction = "attack.accuracy.lock-salvo";

	public const string SpendContainAction = "defense.ship.spend-contain";

	public const string SpendScatterAction = "defense.ship.spend-scatter";

	public const string MajorDerlinAction = "defense.upgrade.major-derlin.reduce-damage";

	public const string ForemansLaborAction = "defense.upgrade.foremans-labor.reduce-damage";

	public const string ElectronicCountermeasuresActionPrefix = "defense.upgrade.electronic-countermeasures.";

	public const string LeadingShotsAction = "attack.dice.leading-shots";

	public const string CaitkenAndShollanActionPrefix = "attack.dice.caitken-and-shollan.";

	public const string OrdnanceExpertsAction = "attack.dice.ordnance-experts";

	public const string AdmiralScreedAction = "attack.dice.admiral-screed";

	public const string WarlordAction = "attack.dice.warlord";

	public const string H9TurbolasersAction = "attack.dice.h9-turbolasers";

	public const string HomeOneAction = "attack.dice.home-one";

	public const string VeteranGunnersAction = "attack.dice.veteran-gunners";

	public const string SensorTeamAction = "attack.dice.sensor-team";

	public const string WeaponsBatteryTechsAction = "attack.dice.weapons-battery-techs";

	public const string WeaponsBatteryTechsActionPrefix = "attack.dice.weapons-battery-techs.";

	public const string LinkedTurbolaserTowersAction = "attack.dice.linked-turbolaser-towers";

	public const string DualTurbolaserTurretsAction = "attack.dice.dual-turbolaser-turrets";

	public const string TurbolaserRerouteCircuitsCriticalAction = "attack.dice.turbolaser-reroute-circuits.critical";

	public const string TurbolaserRerouteCircuitsDoubleHitAction = "attack.dice.turbolaser-reroute-circuits.double-hit";

	public const string ThermalShieldsAction = "defense.upgrade.thermal-shields";

	public const string PointDefenseIonCannonsAction = "defense.upgrade.point-defense-ion-cannons";

	public const string LandoDefenseAction = "defense.upgrade.lando-calrissian";

	public const string BrunsonDefenseAction = "defense.upgrade.captain-brunson";

	public const string ExpertShieldTechAction = "defense.upgrade.expert-shield-tech";

	public const string AdmonitionActionPrefix = "defense.upgrade.admonition.";

	public const string SaNalaorActionPrefix = "defense.upgrade.sa-nalaor.";

	public const string SpendSalvoAction = "defense.ship.spend-salvo";

	public const string ReactiveGunneryActionPrefix = "defense.upgrade.reactive-gunnery.";

	public const string TargetingScramblerAction = "defense.upgrade.targeting-scrambler";

	public const string DefianceActionPrefix = "attack.dice.defiance.";

	public const string DominatorActionPrefix = "attack.dice.dominator.";

	public const string PatriotFistAction = "attack.dice.patriot-fist";

	public const string AvengerAction = "attack.before.avenger";

	public const string GovernorPryceAction = "attack.dice.governor-pryce";

	public const string IntelOfficerActionPrefix = "attack.dice.intel-officer.";

	public const string IntelOfficerSkipAction = "attack.dice.intel-officer.skip";

	public const string DirectorKrennicAction = "attack.dice.director-krennic";

	public const string VaderCommanderAction = "attack.dice.vader-commander";

	public const string VaderOfficerAction = "attack.dice.vader-officer";

	public const string KrakenCommanderAction = "attack.dice.kraken";

	public const string CloneCaptainZakActionPrefix = "attack.dice.clone-captain-zak.";

	public const string GunneryChiefVarnillianAction = "attack.dice.gunnery-chief-varnillian";

	public const string TaskForceOrganaAction = "attack.dice.task-force-organa";

	public const string DisposableCapacitorsAction = "activation.disposable-capacitors";

	public const string PhylonQ7Action = "activation.phylon-q7-tractor-beams";

	public const string PhylonQ7TargetActionPrefix = "activation.phylon-q7-tractor-beams.target.";

	public const float PhylonQ7MaximumDistance = 13.260394f;

	public const string SwivelMountActionPrefix = "command.swivel-mount-batteries.";

	public const string JainaLight = "jainas-light-title";

	public const string Instigator = "instigator-title";

	public const string OrdnanceExperts = "ordnance-experts";

	public const string ExternalRacks = "external-racks";

	public const string B2RocketTroopers = "b2-rocket-troopers";

	public const string CaptainRex = "captain-rex-officer";

	public const string IdenVersio = "iden-versio-officer";

	public const string JynErso = "jyn-erso-boarding-team";

	public const string CommanderBeck = "commander-beck-officer";

	public const string JediHostage = "jedi-hostage";

	public const string SlicerTools = "slicer-tools";

	public const string AuxiliaryShieldsTeam = "auxiliary-shields-team";

	public const int TranquilityMaximumShields = 6;

	public const string Salvation = "salvation-title";

	public const string HondoOhnakaOfficer = "hondo-ohnaka-officer";

	public const string GeneralDodonna = "general-dodonna-commander";

	public const string Amity = "amity-title";

	public const string Concord = "concord-title";

	public const string Endeavor = "endeavor-title";

	public const string Tb73 = "tb-73-title";

	public const string Unity = "unity-title";

	public const string Executor = "executor-title";

	public const string PhoenixHome = "phoenix-home-title";

	public const string Relentless = "relentless-title";

	public const string NovaDefiant = "nova-defiant-title";

	public const string NavigationOfficer = "navigation-officer";

	public const string EngineeringCaptain = "engineering-captain-officer";

	public const string TacticalExpert = "tactical-expert-officer";

	public const string WingCommander = "wing-commander-officer";

	public const string RaymusAntilles = "raymus-antilles-officer";

	public const string VeteranCaptain = "veteran-captain-officer";

	public const string WullfYularen = "wullf-yularen-officer";

	public const string DefenseLiaison = "defense-liaison-officer";

	public const string WeaponsLiaison = "weapons-liaison-officer";

	public const string SkilledFirstOfficer = "skilled-first-officer";

	public const string Fb88 = "fb-88-title";

	public const string WalexBlissex = "walex-blissex-officer";

	public const string TaskmasterGrint = "taskmaster-grint-officer";

	public const string TSeriesTacticalDroid = "t-series-tactical-droid-officer";

	public const string SupportOfficer = "support-officer";

	public const string Pursuant = "pursuant-title";

	public const string CaptainNeeda = "captain-needa-officer";

	public const string Vanguard = "vanguard-title";

	public const string LocalFireControl = "local-fire-control";

	public const string KyrstaAgateCommander = "kyrsta-agate-commander";

	public const string LeadingShots = "leading-shots";

	public const string CaitkenAndShollan = "caitken-and-shollan";

	public const string Warlord = "warlord-title";

	public const string QuadTurbolaserCannons = "quad-turbolaser-cannons";

	public const string H9Turbolasers = "h9-turbolasers";

	public const string VeteranGunners = "veteran-gunners";

	public const string SensorTeam = "sensor-team";

	public const string WeaponsBatteryTechs = "weapons-battery-techs";

	public const string LinkedTurbolaserTowers = "linked-turbolaser-towers";

	public const string DualTurbolaserTurrets = "dual-turbolaser-turrets";

	public const string TurbolaserRerouteCircuits = "turbolaser-reroute-circuits";

	public const string Defiance = "defiance-title";

	public const string Dominator = "dominator-title";

	public const string PatriotFist = "patriot-fist-title";

	public const string Avenger = "avenger-title";

	public const string GovernorPryce = "governor-pryce-officer";

	public const string IntelOfficer = "intel-officer";

	public const string DirectorKrennic = "director-krennic-officer";

	public const string VaderOfficer = "darth-vader-officer";

	public const string CloneCaptainZak = "clone-captain-zak-officer";

	public const string GunneryChiefVarnillian = "gunnery-chief-varnillian";

	public const string TaskForceOrgana = "task-force-organa-title";

	public const string DisposableCapacitors = "disposable-capacitors";

	public const string PhylonQ7TractorBeams = "phylon-q7-tractor-beams";

	public const string SwivelMountBatteries = "swivel-mount-batteries";

	private static readonly Dictionary<string, string> SquadronKeywordText = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
	{
		{ "ADEPT", "While attacking, this squadron may reroll up to {X} attack {DICE}." },
		{ "DODGE", "While defending against a squadron, during the Spend Defense Tokens step, this squadron may choose up to {X} attack dice to be rerolled." },
		{ "SCREEN", "While defending against a squadron, treat this squadron as having Dodge X, where X is the number of other friendly non-Screen squadrons engaged with the attacker, to a maximum of 3." },
		{ "AI ANTISQUADRON", "While attacking with its anti-squadron armament, if activated by a Squadron command, this squadron may add {X} {DICE} to its attack pool of a color or colors already in its attack pool." },
		{ "AI BATTERY", "While attacking with its battery armament, if activated by a Squadron command, this squadron may add {X} {DICE} to its attack pool of a color or colors already in its attack pool." },
		{ "BOMBER", "Critical hits from this squadron damage ships and can trigger critical effects." },
		{ "ESCORT", "Enemy fighters tangled up with this squadron must shoot at it first if they can." },
		{ "SWARM", "Rerolls a die when attacking a squadron that is already tied up with another friendly squadron." },
		{ "ROGUE", "Acts independently - can move and attack in either order without waiting for a ship's orders." },
		{ "HEAVY", "Does not pin enemy fighters in place; they can still move and attack ships while tangled with it." },
		{ "GRIT", "Can still break away and move while only one enemy squadron is tangled up with it." },
		{ "INTEL", "Friendly squadrons close to this one can break away from a single enemy the same way (grants Grit)." },
		{ "CLOAK", "Slips away at the end of the fighter phase, moving a short distance even while tangled up." },
		{ "STRATEGIC", "Can drag an objective token along with it when it finishes moving nearby." },
		{ "ASSAULT", "Can spend a die when attacking a ship to saddle it with a raid token, disrupting its orders." },
		{ "COUNTER", "Shoots back with {X} dice when attacked by another squadron - even if this squadron is destroyed." },
		{ "SNIPE", "Attacks squadrons from a distance with {X} dice, out of reach of return fire." },
		{ "RELAY", "Extends a ship's fighter control: up to {X} squadrons it orders may be far from the ship but near this one." },
		{ "ACE", "A unique squadron. Only one with this unique name may be fielded in a fleet." }
	};

	private static readonly IReadOnlyDictionary<string, DefenseTokenType> FixedSetupDefenseTokens = new Dictionary<string, DefenseTokenType>(StringComparer.OrdinalIgnoreCase)
	{
		{
			"amity-title",
			DefenseTokenType.Evade
		},
		{
			"concord-title",
			DefenseTokenType.Salvo
		},
		{
			"endeavor-title",
			DefenseTokenType.Contain
		},
		{
			"tb-73-title",
			DefenseTokenType.Evade
		},
		{
			"unity-title",
			DefenseTokenType.Redirect
		}
	};

	public static IReadOnlyList<PassiveEffectNotice> ApplyFixedSetupEffects(ShipView ship)
	{
		List<PassiveEffectNotice> list = new List<PassiveEffectNotice>();
		if (ship?.Loadout == null)
		{
			return list;
		}
		foreach (KeyValuePair<string, DefenseTokenType> fixedSetupDefenseToken in FixedSetupDefenseTokens)
		{
			if (ship.Loadout.HasActive(fixedSetupDefenseToken.Key))
			{
				ship.State.AddDefenseToken(fixedSetupDefenseToken.Value);
				UpgradeCardRuntimeState upgradeCardRuntimeState = ship.Loadout.Find(fixedSetupDefenseToken.Key);
				list.Add(PassiveEffectNotice.LogOnly("[SETUP] " + (upgradeCardRuntimeState?.Definition.name ?? fixedSetupDefenseToken.Key) + ": " + ship.Definition.DisplayName + " gains one " + fixedSetupDefenseToken.Value.ToString() + " defense token."));
			}
		}
		if (ship.Loadout.HasActive("executor-title"))
		{
			ship.Commands?.ConfigureTokenStorage(int.MaxValue, allowDuplicates: true);
			list.Add(PassiveEffectNotice.LogOnly("[SETUP] Executor removes the command-token capacity and duplicate-type limits."));
		}
		if (ship.Loadout.HasActive("phoenix-home-title"))
		{
			ship.Commands?.ConfigureTokenStorage(4);
			list.Add(PassiveEffectNotice.LogOnly("[SETUP] Phoenix Home raises command-token capacity to 4."));
		}
		if (ship.Loadout.HasActive("relentless-title") && ship.Commands != null)
		{
			ship.Commands.ConfigureCommandCapacity(ship.Commands.Capacity - 1);
			list.Add(PassiveEffectNotice.LogOnly("[SETUP] Relentless reduces the required command stack by 1 dial."));
		}
		if (ship.Loadout.HasActive("nova-defiant-title"))
		{
			ship.Commands?.ConfigureCommandCapacity(4);
			ship.Commands?.ConfigureTokenStorage(4, allowDuplicates: true);
			list.Add(PassiveEffectNotice.LogOnly("[SETUP] Nova Defiant sets command capacity to 4 and allows duplicate command tokens."));
		}
		return list;
	}

	public static string GetKeywordRulesText(string keyword)
	{
		if (string.IsNullOrWhiteSpace(keyword))
		{
			return string.Empty;
		}
		string text = keyword.Trim();
		string text2 = string.Empty;
		int num = text.LastIndexOf(' ');
		if (num > 0 && int.TryParse(text.Substring(num + 1), out var result))
		{
			text2 = result.ToString();
			text = text.Substring(0, num);
		}
		if (!SquadronKeywordText.TryGetValue(text, out var value))
		{
			return "No summary on file for this keyword yet - check the squadron card.";
		}
		string newValue = (string.IsNullOrEmpty(text2) ? "several" : text2);
		return value.Replace("{X}", newValue).Replace("{DICE}", (text2 == "1") ? "die" : "dice");
	}

	public static bool AttacksCannotBeObstructed(ShipView ship)
	{
		if ((object)ship == null)
		{
			return false;
		}
		return ship.Loadout?.HasActive("jainas-light-title") == true;
	}

	public static bool IgnoresObstacleOverlap(ShipView ship)
	{
		if ((object)ship == null)
		{
			return false;
		}
		return ship.Loadout?.HasActive("jainas-light-title") == true;
	}

	public static int GetBlackAttackRerolls(ShipView ship)
	{
		if ((object)ship == null || ship.Loadout?.HasActive("ordnance-experts") != true)
		{
			return 0;
		}
		return 2;
	}

	public static int ConsumeExternalRacks(ShipView ship, RangeBand range)
	{
		if (range != RangeBand.Close)
		{
			return 0;
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState = ship?.Loadout?.Find("external-racks");
		if (upgradeCardRuntimeState == null || !upgradeCardRuntimeState.Discard())
		{
			return 0;
		}
		return 2;
	}

	public static bool CanUseExternalRacks(ShipView ship, RangeBand range)
	{
		if (range == RangeBand.Close)
		{
			if ((object)ship == null)
			{
				return false;
			}
			return ship.Loadout?.Find("external-racks")?.Active == true;
		}
		return false;
	}

	public static int UseExternalRacks(ShipView ship, RangeBand range)
	{
		return ConsumeExternalRacks(ship, range);
	}

	public static int GetCriticalDamagePerIcon(ShipView ship, HullZone attackZone)
	{
		if ((object)ship == null || ship.Loadout?.HasActive("salvation-title") != true || attackZone != HullZone.Front)
		{
			return 1;
		}
		return 2;
	}

	public static bool HasAuxiliaryShieldsTeam(ShipView ship)
	{
		if ((object)ship == null)
		{
			return false;
		}
		return ship.Loadout?.HasActive("auxiliary-shields-team") == true;
	}

	public static int GetEffectiveShieldMaximum(ShipView ship, HullZone zone)
	{
		int valueOrDefault = (ship?.State?.GetMaximumShields(zone)).GetValueOrDefault();
		if (!HasAuxiliaryShieldsTeam(ship) || (zone != HullZone.Left && zone != HullZone.Right))
		{
			return valueOrDefault;
		}
		return Math.Min(4, valueOrDefault + 1);
	}

	public static int GetDisplayedShieldMaximum(ShipView ship, HullZone zone, int observedMinimum = 0)
	{
		return Math.Max(GetEffectiveShieldMaximum(ship, zone), Math.Max(observedMinimum, (ship?.State?.GetShields(zone)).GetValueOrDefault()));
	}

	public static bool DiscardUpgradeFromShip(ShipView ship, UpgradeCardRuntimeState upgrade, out int shieldsLost)
	{
		shieldsLost = 0;
		if (ship?.Loadout == null || upgrade == null || !upgrade.Active)
		{
			return false;
		}
		bool flag = false;
		foreach (UpgradeCardRuntimeState upgrade2 in ship.Loadout.Upgrades)
		{
			if (upgrade2 == upgrade)
			{
				flag = true;
				break;
			}
		}
		if (!flag)
		{
			return false;
		}
		bool flag2 = string.Equals(upgrade.Definition.id, "auxiliary-shields-team", StringComparison.OrdinalIgnoreCase);
		if (!upgrade.Discard())
		{
			return false;
		}
		if (!flag2)
		{
			return true;
		}
		HullZone[] array = new HullZone[2]
		{
			HullZone.Left,
			HullZone.Right
		};
		foreach (HullZone zone in array)
		{
			int amount = Math.Max(0, ship.State.GetShields(zone) - ship.State.GetMaximumShields(zone));
			shieldsLost += ship.State.LoseShields(zone, amount);
		}
		return true;
	}

	public static bool IsVirtuallyEngagedByInstigator(SquadronView squadron)
	{
		if (squadron == null || squadron.IsDestroyed)
		{
			return false;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId != squadron.OwnerId)
			{
				ShipLoadoutState loadout = shipView.Loadout;
				if (loadout != null && loadout.HasActive("instigator-title") && SquadronAbilityEngine.FootprintDistance(squadron, shipView) <= 3.369803f)
				{
					return true;
				}
			}
		}
		return false;
	}

	public static IReadOnlyList<AbilityAction> GetActions(ShipView ship, AbilityTiming timing)
	{
		return GetActions(ship, new AbilityQueryContext(timing));
	}

	public static IReadOnlyList<AbilityAction> GetActions(ShipView ship, AbilityQueryContext context)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (ship?.Loadout == null)
		{
			return list;
		}
		foreach (UpgradeCardRuntimeState upgrade in ship.Loadout.Upgrades)
		{
			if (!upgrade.Active)
			{
				continue;
			}
			if (context.Timing == AbilityTiming.StartOfActivation && upgrade.Definition.id == "interdictor-title" && !upgrade.Exhausted)
			{
				foreach (UpgradeCardRuntimeState upgrade2 in ship.Loadout.Upgrades)
				{
					if (upgrade2 != upgrade && upgrade2.Active && upgrade2.Exhausted)
					{
						list.Add(new AbilityAction("remaining.start-activation.interdictor." + upgrade2.Definition.id, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "READY " + upgrade2.Definition.name.ToUpperInvariant(), "Exhaust Interdictor to ready this equipped card.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					}
				}
				continue;
			}
			if (context.Timing == AbilityTiming.StartOfActivation && upgrade.Definition.id == "beast-of-burden-title" && !upgrade.Exhausted && HasReadyDefenseToken(ship))
			{
				foreach (ShipView item in FriendlyShipsWithSpentDefenseToken(ship, 8.227571f))
				{
					foreach (DefenseTokenType value3 in Enum.GetValues(typeof(DefenseTokenType)))
					{
						if (ship.State.GetReadyDefenseTokenCount(value3) <= 0)
						{
							continue;
						}
						foreach (DefenseTokenType value4 in Enum.GetValues(typeof(DefenseTokenType)))
						{
							if (item.State.GetAvailableDefenseTokenCount(value4) > item.State.GetReadyDefenseTokenCount(value4))
							{
								string[] obj = new string[6]
								{
									"remaining.start-activation.beast-of-burden.target.",
									item.RuntimeId.Value,
									".spend.",
									null,
									null,
									null
								};
								int num = (int)value3;
								obj[3] = num.ToString();
								obj[4] = ".ready.";
								num = (int)value4;
								obj[5] = num.ToString();
								list.Add(new AbilityAction(string.Concat(obj), upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "BEAST: " + item.Definition.DisplayName.ToUpperInvariant(), "Spend " + value3.ToString() + " to ready this ship's " + value4.ToString() + " token.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing, optional: true, selected: false, item.RuntimeId));
							}
						}
					}
				}
				continue;
			}
			if (context.Timing == AbilityTiming.StartOfActivation && upgrade.Definition.id == "hand-of-justice-title" && !upgrade.Exhausted)
			{
				foreach (ShipView item2 in FriendlyShipsWithSpentDefenseToken(ship, 13.260394f))
				{
					foreach (DefenseTokenType value5 in Enum.GetValues(typeof(DefenseTokenType)))
					{
						if (item2.State.GetAvailableDefenseTokenCount(value5) > item2.State.GetReadyDefenseTokenCount(value5))
						{
							string value = item2.RuntimeId.Value;
							int num = (int)value5;
							list.Add(new AbilityAction("remaining.start-activation.hand-of-justice.target." + value + ".ready." + num, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "HAND: " + item2.Definition.DisplayName.ToUpperInvariant(), "Exhaust this title to ready this ship's " + value5.ToString() + " token.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing, optional: true, selected: false, item2.RuntimeId));
						}
					}
				}
				continue;
			}
			if (context.Timing == AbilityTiming.StartOfActivation && upgrade.Definition.id == "squall-title" && HasEligibleSquallTarget(ship))
			{
				list.Add(new AbilityAction("remaining.start-activation.squall", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "SQUALL: REPOSITION FIGHTERS", "Choose up to three unengaged friendly squadrons at close-medium range. Move each up to distance 2 without ending engaged.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				continue;
			}
			if (context.Timing == AbilityTiming.CommandReveal && RemainingUpgradeRules.IsBoardingTeam(upgrade.Definition.id))
			{
				if (context.RevealedCommand != ShipCommand.Squadron)
				{
					CommandStackState commands = ship.Commands;
					if (commands == null || !commands.HasToken(ShipCommand.Squadron))
					{
						goto IL_06fc;
					}
				}
				if (HasEnemyShipInRange(ship, 5.426696f))
				{
					ShipView[] array = GameScope.Current.SceneShipViews;
					foreach (ShipView shipView in array)
					{
						if (!(shipView == null) && !shipView.IsDestroyed && shipView.OwnerId != ship.OwnerId && shipView.RuntimeId.IsValid && !(RaidRules.FootprintDistance(ship, shipView) > 5.426696f))
						{
							list.Add(new AbilityAction("remaining.command-reveal.boarding." + upgrade.Definition.id + "." + shipView.RuntimeId.Value, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "BOARD " + shipView.Definition.DisplayName.ToUpperInvariant(), "Spend a Squadron dial or token, discard " + upgrade.Definition.name + ", and choose this close-range enemy ship.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing, optional: true, selected: false, shipView.RuntimeId));
						}
					}
					continue;
				}
			}
			goto IL_06fc;
			IL_06fc:
			if (context.Timing == AbilityTiming.CommandReveal && upgrade.Definition.id == "ahsoka-tano-officer-rebel" && !upgrade.Exhausted)
			{
				CommandStackState commands2 = ship.Commands;
				if (commands2 != null && commands2.TokenCount > 0)
				{
					foreach (ShipCommand value6 in Enum.GetValues(typeof(ShipCommand)))
					{
						CommandStackState commands3 = ship.Commands;
						if (commands3 != null && commands3.CanStoreToken(value6))
						{
							int num = (int)value6;
							list.Add(new AbilityAction("remaining.command-reveal.ahsoka." + num, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "AHSOKA: GAIN " + value6.ToString().ToUpperInvariant(), "Exhaust Ahsoka, discard another token, and gain this command token.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
						}
					}
					continue;
				}
			}
			if (context.Timing == AbilityTiming.CommandReveal && context.RevealedCommand == ShipCommand.Squadron && upgrade.UsesThisActivation == 0 && upgrade.Definition.id == "clone-commander-wolffe-officer" && !upgrade.Exhausted)
			{
				list.Add(new AbilityAction("remaining.command-reveal.squadron-keyword." + upgrade.Definition.id, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "GRANT ASSAULT", "Apply this officer's temporary keyword to up to three squadrons activated by this order.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				continue;
			}
			if (upgrade.Definition.id == "disposable-capacitors" && context.Timing == AbilityTiming.StartOfActivation)
			{
				list.Add(new AbilityAction("activation.disposable-capacitors", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "DISCARD DISPOSABLE CAPACITORS", "Use blue battery dice at long range for this activation.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				continue;
			}
			if (upgrade.Definition.id == "phylon-q7-tractor-beams" && context.Timing == AbilityTiming.StartOfActivation && !upgrade.Exhausted && LegalPhylonTargets(ship).Any())
			{
				list.Add(new AbilityAction("activation.phylon-q7-tractor-beams", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "PHYLON Q7 TRACTOR BEAMS", "Enter Phylon Q7 targeting, then choose an enemy ship at distance 1-5.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				continue;
			}
			if (upgrade.Definition.id == "swivel-mount-batteries" && context.Timing == AbilityTiming.CommandReveal && !upgrade.Exhausted)
			{
				foreach (HullZone hullZone in ship.Definition.HullZones)
				{
					int num = (int)hullZone;
					list.Add(new AbilityAction("command.swivel-mount-batteries." + num, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "FOCUS " + hullZone.ToString().ToUpperInvariant(), "Exhaust Swivel-Mount Batteries and focus this hull zone.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				}
				continue;
			}
			if (upgrade.Definition.id == "external-racks" && context.Timing == AbilityTiming.BeforeAttack && CanUseExternalRacks(ship, context.Range))
			{
				list.Add(new AbilityAction("upgrade.external-racks.add-black-dice", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "USE EXTERNAL RACKS", "Discard External Racks to add two black weapon dice to this close-range attack.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				continue;
			}
			if (upgrade.Definition.id == "b2-rocket-troopers" && context.Timing == AbilityTiming.BeforeAttack && !upgrade.Exhausted && (context.Range == RangeBand.Close || context.Range == RangeBand.Medium))
			{
				list.Add(new AbilityAction("upgrade.b2-rocket-troopers.place-raid", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "DEPLOY B2 ROCKET TROOPERS", "Exhaust B2 Rocket Troopers before a close- or medium-range attack against a ship to place a raid token. It is not available during anti-squadron attacks.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				continue;
			}
			if (upgrade.Definition.id == "iden-versio-officer" && context.Timing == AbilityTiming.CommandReveal && context.RevealedCommand == ShipCommand.Squadron && HasEnemyShipInRange(ship, 5.426696f))
			{
				list.Add(new AbilityAction("upgrade.iden-versio.place-raid", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "IDEN VERSIO: RAID", "Discard Iden Versio to raid an enemy ship at close range.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				continue;
			}
			if (upgrade.Definition.id == "jyn-erso-boarding-team" && context.Timing == AbilityTiming.CommandReveal && HasEnemyShipInRange(ship, 5.426696f, requireUnraided: true))
			{
				if (context.RevealedCommand == ShipCommand.Squadron)
				{
					list.Add(new AbilityAction("upgrade.jyn-erso.place-raid-with-dial", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "JYN ERSO: SPEND DIAL", "Discard this Squadron dial and Jyn Erso to place two raid tokens on an unraided enemy ship at close range.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				}
				CommandStackState commands4 = ship.Commands;
				if (commands4 != null && commands4.HasToken(ShipCommand.Squadron))
				{
					list.Add(new AbilityAction("upgrade.jyn-erso.place-raid-with-token", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "JYN ERSO: SPEND TOKEN", "Spend a Squadron token and discard Jyn Erso to place two raid tokens on an unraided enemy ship at close range.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
				}
				continue;
			}
			if (upgrade.Definition.id == "commander-beck-officer" && context.Timing == AbilityTiming.DuringActivation && upgrade.UsesThisActivation < 2)
			{
				foreach (ShipCommand value7 in Enum.GetValues(typeof(ShipCommand)))
				{
					CommandStackState commands5 = ship.Commands;
					if (commands5 != null && commands5.CanGainRaidToken(value7))
					{
						int num = (int)value7;
						list.Add(new AbilityAction("upgrade.commander-beck.resolve." + num, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "BECK: " + value7.ToString().ToUpperInvariant(), "Resolve this command at token strength, then take its matching raid token.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					}
				}
				continue;
			}
			if (context.Timing == AbilityTiming.StartOfActivation && TryGetFixedCommandChange(upgrade.Definition.id, out var command))
			{
				CommandStackState commands6 = ship.Commands;
				if (commands6 != null && commands6.TryPeekTop(out var command2) && command2 != command)
				{
					int num = (int)command;
					list.Add(new AbilityAction("command.change." + num + "." + upgrade.Definition.id, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "CHANGE TO " + command.ToString().ToUpperInvariant(), "Change the top command dial to " + command.ToString() + " before revealing it.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					continue;
				}
			}
			if (upgrade.Definition.id == "veteran-captain-officer" && context.Timing == AbilityTiming.CommandReveal)
			{
				foreach (ShipCommand value8 in Enum.GetValues(typeof(ShipCommand)))
				{
					CommandStackState commands7 = ship.Commands;
					if (commands7 != null && commands7.CanStoreToken(value8))
					{
						int num = (int)value8;
						list.Add(new AbilityAction("command.veteran-captain.gain." + num, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "GAIN " + value8.ToString().ToUpperInvariant(), "Discard Veteran Captain to gain this command token.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					}
				}
				continue;
			}
			if (upgrade.Definition.id == "raymus-antilles-officer" && context.Timing == AbilityTiming.CommandReveal && upgrade.UsesThisActivation == 0 && context.RevealedCommand.HasValue)
			{
				CommandStackState commands8 = ship.Commands;
				if (commands8 != null && commands8.CanStoreToken(context.RevealedCommand.Value))
				{
					list.Add(new AbilityAction("command.raymus-antilles.gain-matching", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "GAIN MATCHING TOKEN", "Gain a " + context.RevealedCommand.Value.ToString() + " token without spending the dial.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					continue;
				}
			}
			if (upgrade.Definition.id == "taskmaster-grint-officer" && context.Timing == AbilityTiming.CommandReveal && upgrade.UsesThisActivation == 0 && context.RevealedCommand.HasValue && upgrade.AssignedCommandToken == context.RevealedCommand)
			{
				CommandStackState commands9 = ship.Commands;
				if (commands9 != null && commands9.CanStoreToken(context.RevealedCommand.Value))
				{
					list.Add(new AbilityAction("command.taskmaster-grint.gain-matching", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "GRINT: GAIN MATCHING TOKEN", "Gain a matching command token without spending the revealed dial.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					continue;
				}
			}
			if (context.Timing == AbilityTiming.CommandReveal && upgrade.UsesThisActivation == 0 && (upgrade.Definition.id == "defense-liaison-officer" || upgrade.Definition.id == "weapons-liaison-officer"))
			{
				ShipCommand[] array2 = ((!(upgrade.Definition.id == "defense-liaison-officer")) ? new ShipCommand[2]
				{
					ShipCommand.ConcentrateFire,
					ShipCommand.Squadron
				} : new ShipCommand[2]
				{
					ShipCommand.Navigate,
					ShipCommand.Engineering
				});
				foreach (ShipCommand token in ship.Commands.Tokens)
				{
					ShipCommand[] array3 = array2;
					for (int num = 0; num < array3.Length; num++)
					{
						ShipCommand shipCommand4 = array3[num];
						if (context.RevealedCommand != shipCommand4)
						{
							string[] obj2 = new string[6] { "command.liaison.", null, null, null, null, null };
							int num2 = (int)shipCommand4;
							obj2[1] = num2.ToString();
							obj2[2] = ".";
							num2 = (int)token;
							obj2[3] = num2.ToString();
							obj2[4] = ".";
							obj2[5] = upgrade.Definition.id;
							list.Add(new AbilityAction(string.Concat(obj2), upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "SPEND " + token.ToString().ToUpperInvariant() + " → " + shipCommand4.ToString().ToUpperInvariant(), "Spend the indicated command token to change the revealed command.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
						}
					}
				}
				continue;
			}
			if (context.Timing == AbilityTiming.StartOfActivation && upgrade.UsesThisActivation == 0)
			{
				CommandStackState commands10 = ship.Commands;
				if (commands10 != null && commands10.Count > 0 && (upgrade.Definition.id == "skilled-first-officer" || upgrade.Definition.id == "fb-88-title"))
				{
					list.Add(new AbilityAction("command.discard-top." + upgrade.Definition.id, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "DISCARD TOP COMMAND", (upgrade.Definition.id == "skilled-first-officer") ? "Discard Skilled First Officer and the top command dial before revealing." : "Discard the top command dial before revealing.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					continue;
				}
			}
			if (context.Timing == AbilityTiming.StartOfActivation && upgrade.Definition.id == "walex-blissex-officer")
			{
				foreach (DefenseTokenType value9 in Enum.GetValues(typeof(DefenseTokenType)))
				{
					if (ship.State.GetDiscardedDefenseTokenCount(value9) > 0)
					{
						int num = (int)value9;
						list.Add(new AbilityAction("defense.walex-blissex.recover." + num, upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "RECOVER " + value9.ToString().ToUpperInvariant(), "Discard Walex Blissex to recover this discarded defense token.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					}
				}
				continue;
			}
			if (context.Timing == AbilityTiming.StartOfRound && upgrade.Definition.id == "support-officer")
			{
				CommandStackState commands11 = ship.Commands;
				if (commands11 != null && commands11.Count > 0)
				{
					list.Add(new AbilityAction("command.support-officer.discard-stack", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "DISCARD COMMAND STACK", "Discard Support Officer and all assigned command dials, then program this ship's full stack.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
					continue;
				}
			}
			if (context.Timing == AbilityTiming.CommandReveal && upgrade.Definition.id == "pursuant-title" && context.RevealedCommand.HasValue && context.RevealedCommand != ShipCommand.Squadron)
			{
				list.Add(new AbilityAction("command.pursuant.resolve-squadron", upgrade.Definition.id, upgrade.Definition.name, AbilitySourceKind.Upgrade, "DISCARD PURSUANT: SQUADRON", "Discard Pursuant to resolve a Squadron command at dial strength instead of the revealed command.", RulesTextPresenter.ToVideoGameTerms(upgrade.Definition.ability), context.Timing));
			}
			else if (!(upgrade.Definition.id == "jainas-light-title"))
			{
				_ = upgrade.Definition.id == "instigator-title";
			}
		}
		if (context.Timing == AbilityTiming.CommandReveal && context.RevealedCommand.HasValue)
		{
			CommandStackState commands12 = ship.Commands;
			if (commands12 != null && commands12.HasToken(context.RevealedCommand.Value))
			{
				ShipCommand value2 = context.RevealedCommand.Value;
				UpgradeCardRuntimeState upgradeCardRuntimeState = ship.Loadout.Find("t-series-tactical-droid-officer");
				if (value2 != ShipCommand.ConcentrateFire && upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && !upgradeCardRuntimeState.Exhausted)
				{
					list.Add(new AbilityAction("command.token-as-dial.t-series-tactical-droid-officer", "t-series-tactical-droid-officer", upgradeCardRuntimeState.Definition.name, AbilitySourceKind.Upgrade, "T-SERIES: TOKEN AS DIAL", "Spend the matching command token and exhaust T-series Tactical Droid to resolve it at dial strength.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState.Definition.ability), context.Timing));
				}
				UpgradeCardRuntimeState upgradeCardRuntimeState2 = CommanderRules.FindFleetUpgrade(ship, "admiral-piett-commander");
				if (upgradeCardRuntimeState2 != null && upgradeCardRuntimeState2.Active && !upgradeCardRuntimeState2.Exhausted)
				{
					list.Add(new AbilityAction("command.token-as-dial.admiral-piett-commander", "admiral-piett-commander", upgradeCardRuntimeState2.Definition.name, AbilitySourceKind.Upgrade, "PIETT: TOKEN AS DIAL", "Spend the matching command token and exhaust Admiral Piett to resolve it at dial strength.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState2.Definition.ability), context.Timing));
				}
			}
		}
		return list;
	}

	private static bool HasEligibleSquallTarget(ShipView ship)
	{
		if (ship == null)
		{
			return false;
		}
		SquadronView[] array = GameScope.Current.SceneSquadronViews;
		SquadronView[] array2 = array;
		foreach (SquadronView squadronView in array2)
		{
			if (squadronView == null || squadronView.IsDestroyed || squadronView.OwnerId != ship.OwnerId || squadronView.State.TemporaryAttackOnly || SquadronAbilityEngine.FootprintDistance(squadronView, ship) > 8.227571f)
			{
				continue;
			}
			bool flag = false;
			SquadronView[] array3 = array;
			foreach (SquadronView second in array3)
			{
				if (SquadronAbilityEngine.IsEngaged(squadronView, second))
				{
					flag = true;
					break;
				}
			}
			if (!flag)
			{
				return true;
			}
		}
		return false;
	}

	private static bool HasReadyDefenseToken(ShipView ship)
	{
		if (ship?.State == null)
		{
			return false;
		}
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			if (ship.State.GetReadyDefenseTokenCount(value) > 0)
			{
				return true;
			}
		}
		return false;
	}

	private static IEnumerable<ShipView> FriendlyShipsWithSpentDefenseToken(ShipView source, float maximumRange)
	{
		if (source == null)
		{
			yield break;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (!(shipView == null) && !(shipView == source) && !shipView.IsDestroyed && shipView.OwnerId == source.OwnerId && shipView.RuntimeId.IsValid && !(RaidRules.FootprintDistance(source, shipView) > maximumRange) && HasSpentDefenseToken(shipView))
			{
				yield return shipView;
			}
		}
	}

	private static bool HasSpentDefenseToken(ShipView ship)
	{
		if (ship?.State == null)
		{
			return false;
		}
		foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
		{
			if (ship.State.GetAvailableDefenseTokenCount(value) > ship.State.GetReadyDefenseTokenCount(value))
			{
				return true;
			}
		}
		return false;
	}

	private static bool HasEnemyShipInRange(ShipView source, float maximumRange, bool requireUnraided = false)
	{
		if (source == null)
		{
			return false;
		}
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (shipView != null && !shipView.IsDestroyed && shipView.OwnerId != source.OwnerId && shipView.Commands != null && (!requireUnraided || shipView.Commands.RaidTokenCount == 0) && RaidRules.FootprintDistance(source, shipView) <= maximumRange)
			{
				return true;
			}
		}
		return false;
	}

	private static IEnumerable<ShipView> LegalPhylonTargets(ShipView source)
	{
		if (source == null)
		{
			yield break;
		}
		int sourceSize = RemainingUpgradeRules.SizeRank(source);
		ShipView[] array = GameScope.Current.SceneShipViews;
		foreach (ShipView shipView in array)
		{
			if (shipView != null && !shipView.IsDestroyed && shipView.OwnerId != source.OwnerId && shipView.RuntimeId.IsValid && RemainingUpgradeRules.SizeRank(shipView) <= sourceSize && RaidRules.FootprintDistance(source, shipView) <= 13.260394f)
			{
				yield return shipView;
			}
		}
	}

	private static bool TryGetFixedCommandChange(string cardId, out ShipCommand command)
	{
		switch (cardId)
		{
		case "navigation-officer":
			command = ShipCommand.Navigate;
			return true;
		case "engineering-captain-officer":
			command = ShipCommand.Engineering;
			return true;
		case "tactical-expert-officer":
			command = ShipCommand.ConcentrateFire;
			return true;
		case "wing-commander-officer":
			command = ShipCommand.Squadron;
			return true;
		default:
			command = ShipCommand.Navigate;
			return false;
		}
	}

	public static IReadOnlyList<AbilityAction> GetShipAccuracyActions(ShipView defender, int accuraciesRemaining, ISet<DefenseTokenType> lockedTokens)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (defender == null || accuraciesRemaining <= 0)
		{
			return list;
		}
		AddShipAccuracyAction(list, defender, DefenseTokenType.Contain, "attack.accuracy.lock-contain", "LOCK CONTAIN", lockedTokens);
		AddShipAccuracyAction(list, defender, DefenseTokenType.Scatter, "attack.accuracy.lock-ship-scatter", "LOCK SCATTER", lockedTokens);
		AddShipAccuracyAction(list, defender, DefenseTokenType.Salvo, "attack.accuracy.lock-salvo", "LOCK SALVO", lockedTokens);
		return list;
	}

	public static IReadOnlyList<AbilityAction> GetGatheredAttackDiceDefenseActions(ShipView defender, int redDice, int blueDice, int blackDice)
	{
		int initialGatheredDiceCount = Math.Max(0, redDice) + Math.Max(0, blueDice) + Math.Max(0, blackDice);
		return GetGatheredAttackDiceDefenseActions(defender, redDice, blueDice, blackDice, initialGatheredDiceCount);
	}

	public static IReadOnlyList<AbilityAction> GetGatheredAttackDiceDefenseActions(ShipView defender, int redDice, int blueDice, int blackDice, int initialGatheredDiceCount)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		int remainingDiceCount = Math.Max(0, redDice) + Math.Max(0, blueDice) + Math.Max(0, blackDice);
		if (!DefenseUpgradeRules.CanUseThermalShields(defender, initialGatheredDiceCount, remainingDiceCount))
		{
			return list;
		}
		int num = DefenseUpgradeRules.ThermalShieldsRemovalCount(initialGatheredDiceCount, remainingDiceCount);
		UpgradeCardRuntimeState upgradeCardRuntimeState = defender.Loadout.Find("thermal-shields");
		list.Add(new AbilityAction("defense.upgrade.thermal-shields", "thermal-shields", upgradeCardRuntimeState.Definition.name, AbilitySourceKind.Upgrade, "THERMAL SHIELDS: REMOVE " + num + " DICE", "Spend one Brace token, then choose and remove half of the gathered attack dice before they are rolled, rounded down.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState.Definition.ability), AbilityTiming.AfterGatherAttackDice));
		return list;
	}

	public static IReadOnlyList<AbilityAction> GetShipDefenseActions(ShipView defender, AttackResolution attack, HullZone defendingZone, ISet<DefenseTokenType> lockedTokens, ISet<DefenseTokenInstance> lockedTokenInstances, ISet<DefenseTokenType> spentThisAttack, bool allowSalvo = true)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (defender == null || defender.IsDestroyed)
		{
			return list;
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState = defender.Loadout?.Find("electronic-countermeasures");
		if (upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && !upgradeCardRuntimeState.Exhausted && lockedTokenInstances != null)
		{
			foreach (DefenseTokenInstance item in from instance in lockedTokenInstances
				orderby instance.Token, instance.Slot
				select instance)
			{
				if ((spentThisAttack == null || !spentThisAttack.Contains(item.Token)) && defender.State.IsDefenseTokenInstanceAvailable(item) && (!item.Exhausted || defender.State.CanSpendExhaustedDefenseTokens))
				{
					list.Add(new AbilityAction(ForcedDefenseTokenDiscardRules.BuildChoiceId("defense.upgrade.electronic-countermeasures.", item), "electronic-countermeasures", upgradeCardRuntimeState.Definition.name, AbilitySourceKind.Upgrade, "ECM: SPEND LOCKED " + item.Token.ToString().ToUpperInvariant() + " #" + (item.Slot + 1), "Exhaust Electronic Countermeasures to spend this accuracy-locked defense token.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState.Definition.ability), AbilityTiming.Defense));
				}
			}
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState2 = defender.Loadout?.Find("point-defense-ion-cannons");
		if (attack.Range == RangeBand.Close && upgradeCardRuntimeState2 != null && upgradeCardRuntimeState2.Active && upgradeCardRuntimeState2.UsesThisAttack == 0)
		{
			list.Add(new AbilityAction("defense.upgrade.point-defense-ion-cannons", "point-defense-ion-cannons", upgradeCardRuntimeState2.Definition.name, AbilitySourceKind.Upgrade, "POINT DEFENSE: FORCE REROLL", "Choose one attack die and force the attacker to reroll it.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState2.Definition.ability), AbilityTiming.Defense));
		}
		ShipView shipView = DefenseUpgradeRules.TargetingScramblerSource(defender);
		if (attack.Range == RangeBand.Close && attack.Dice.Count > 0 && shipView != null)
		{
			UpgradeCardRuntimeState upgradeCardRuntimeState3 = shipView.Loadout.Find("targeting-scrambler");
			list.Add(new AbilityAction("defense.upgrade.targeting-scrambler", "targeting-scrambler", upgradeCardRuntimeState3.Definition.name, AbilitySourceKind.Upgrade, "TARGETING SCRAMBLER: REROLL 4", "Exhaust Targeting Scrambler to force the attacker to reroll up to four attack dice you choose.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState3.Definition.ability), AbilityTiming.Defense));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState4 = defender.Loadout?.Find("lando-calrissian-officer");
		if (upgradeCardRuntimeState4 != null && upgradeCardRuntimeState4.Active && attack.Dice.Count > 0)
		{
			list.Add(new AbilityAction("defense.upgrade.lando-calrissian", "lando-calrissian-officer", upgradeCardRuntimeState4.Definition.name, AbilitySourceKind.Upgrade, "LANDO: CHOOSE REROLLS", "Discard Lando Calrissian to choose one or more attack dice and force them to be rerolled.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState4.Definition.ability), AbilityTiming.Defense));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState5 = defender.Loadout?.Find("captain-brunson-officer");
		if (upgradeCardRuntimeState5 != null && upgradeCardRuntimeState5.Active && !upgradeCardRuntimeState5.Exhausted && OfficerCombatRules.IsNearObstacle(defender))
		{
			list.Add(new AbilityAction("defense.upgrade.captain-brunson", "captain-brunson-officer", upgradeCardRuntimeState5.Definition.name, AbilitySourceKind.Upgrade, "BRUNSON: CANCEL ONE DIE", "Exhaust Captain Brunson to choose and cancel one attack die.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState5.Definition.ability), AbilityTiming.Defense));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState6 = defender.Loadout?.Find("expert-shield-tech-officer");
		if (upgradeCardRuntimeState6 != null && upgradeCardRuntimeState6.Active && upgradeCardRuntimeState6.UsesThisAttack == 0 && (lockedTokens == null || !lockedTokens.Contains(DefenseTokenType.Redirect)) && (spentThisAttack == null || !spentThisAttack.Contains(DefenseTokenType.Redirect)) && defender.State.CanResolveDefenseToken(DefenseTokenType.Redirect))
		{
			list.Add(new AbilityAction("defense.upgrade.expert-shield-tech", "expert-shield-tech-officer", upgradeCardRuntimeState6.Definition.name, AbilitySourceKind.Upgrade, "SHIELD TECH: REDUCE DAMAGE", "Spend Redirect to reduce total damage by one instead of moving damage to another hull zone.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState6.Definition.ability), AbilityTiming.Defense));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState7 = defender.Loadout?.Find("admonition-title");
		if (upgradeCardRuntimeState7 != null && upgradeCardRuntimeState7.Active && upgradeCardRuntimeState7.UsesThisAttack == 0)
		{
			foreach (DefenseTokenInstance availableInstance in ForcedDefenseTokenDiscardRules.GetAvailableInstances(defender.State))
			{
				list.Add(new AbilityAction(ForcedDefenseTokenDiscardRules.BuildChoiceId("defense.upgrade.admonition.", availableInstance), "admonition-title", upgradeCardRuntimeState7.Definition.name, AbilitySourceKind.Upgrade, "ADMONITION: DISCARD " + availableInstance.Token.ToString().ToUpperInvariant() + " #" + (availableInstance.Slot + 1) + (availableInstance.Exhausted ? " | EXHAUSTED" : " | READY"), "Discard this defense token to choose and cancel one attack die.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState7.Definition.ability), AbilityTiming.Defense));
			}
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState8 = defender.Loadout?.Find("sa-nalaor-title");
		if (defender.State.CurrentSpeed >= 2 && upgradeCardRuntimeState8 != null && upgradeCardRuntimeState8.Active && upgradeCardRuntimeState8.UsesThisAttack == 0)
		{
			foreach (DefenseTokenType value in Enum.GetValues(typeof(DefenseTokenType)))
			{
				if ((lockedTokens == null || !lockedTokens.Contains(value)) && (spentThisAttack == null || !spentThisAttack.Contains(value)) && defender.State.CanResolveDefenseToken(value))
				{
					int num = (int)value;
					list.Add(new AbilityAction("defense.upgrade.sa-nalaor." + num, "sa-nalaor-title", upgradeCardRuntimeState8.Definition.name, AbilitySourceKind.Upgrade, "SA NALAOR: " + value.ToString().ToUpperInvariant() + " AS EVADE", "Spend this defense token to resolve the Evade effect instead.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState8.Definition.ability), AbilityTiming.Defense));
				}
			}
		}
		if (defender.State.CurrentSpeed > 0 && attack.CanResolveCriticalEffect)
		{
			AddShipDefenseAction(list, defender, DefenseTokenType.Contain, "defense.ship.spend-contain", "SPEND CONTAIN", "Prevent the standard face-up critical damage card from this attack.", lockedTokens, spentThisAttack);
		}
		if (defender.State.CurrentSpeed > 0)
		{
			AddShipDefenseAction(list, defender, DefenseTokenType.Scatter, "defense.ship.spend-scatter", "SPEND SCATTER", "Cancel every weapon die in this attack and suffer no damage.", lockedTokens, spentThisAttack);
		}
		if (allowSalvo && defender.State.CurrentSpeed > 0)
		{
			AddShipDefenseAction(list, defender, DefenseTokenType.Salvo, "defense.ship.spend-salvo", "SPEND SALVO", "Spend Salvo to return fire after this attack resolves.", lockedTokens, spentThisAttack);
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState9 = defender.Loadout?.Find("reactive-gunnery");
		if (allowSalvo && upgradeCardRuntimeState9 != null && upgradeCardRuntimeState9.Active && !upgradeCardRuntimeState9.Exhausted && upgradeCardRuntimeState9.UsesThisAttack == 0)
		{
			foreach (DefenseTokenType value2 in Enum.GetValues(typeof(DefenseTokenType)))
			{
				if (value2 != DefenseTokenType.Salvo && (lockedTokens == null || !lockedTokens.Contains(value2)) && (spentThisAttack == null || !spentThisAttack.Contains(value2)) && defender.State.CanResolveDefenseToken(value2) && DefenseTokenInstanceChoiceRules.GetSpendableInstances(defender.State, value2, lockedTokenInstances).Count != 0)
				{
					int num = (int)value2;
					list.Add(new AbilityAction("defense.upgrade.reactive-gunnery." + num, "reactive-gunnery", upgradeCardRuntimeState9.Definition.name, AbilitySourceKind.Upgrade, "REACTIVE GUNNERY: " + value2.ToString().ToUpperInvariant(), "Exhaust Reactive Gunnery and spend this token to resolve Salvo instead of its normal effect.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState9.Definition.ability), AbilityTiming.Defense));
				}
			}
		}
		AddDamageReductionAction(list, defender, "major-derlin-officer", "defense.upgrade.major-derlin.reduce-damage", "EXHAUST MAJOR DERLIN", "Exhaust Major Derlin to reduce this attack's total damage by 1.");
		if (defender.State.GetShields(defendingZone) > 0)
		{
			AddDamageReductionAction(list, defender, "foremans-labor-title", "defense.upgrade.foremans-labor.reduce-damage", "EXHAUST FOREMAN'S LABOR", "Exhaust Foreman's Labor to reduce this attack's total damage by 1.");
		}
		return list;
	}

	public static bool HasBombardTrait(ShipView ship)
	{
		if (ship?.Definition != null)
		{
			return ship.Game.Topology.Ships[ship.Index].CatalogEntry.Metadata.Traits.Any((string trait) => string.Equals(trait, "Bombard", StringComparison.OrdinalIgnoreCase));
		}
		return false;
	}

	public static IReadOnlyList<AbilityAction> GetShipDiceActions(ShipView attacker, AttackResolution attack, bool leadingShotsUsed, bool caitkenAndShollanUsed, bool ordnanceExpertsUsed, bool screedUsed, bool warlordUsed, bool homeOneUsed, bool againstSquadron = false)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (attacker?.Loadout == null || attack.Dice == null || attack.Dice.Count == 0)
		{
			return list;
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState = attacker.Loadout.Find("leading-shots");
		if (!leadingShotsUsed && attack.Blanks > 0 && attack.Dice.Count > 1 && upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && AttackResolver.HasDie(attack, AttackDieColor.Blue))
		{
			list.Add(new AbilityAction("attack.dice.leading-shots", "leading-shots", upgradeCardRuntimeState.Definition.name, AbilitySourceKind.Upgrade, "USE LEADING SHOTS", "Choose one blue die to spend, then choose any number of remaining attack dice to reroll.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState2 = attacker.Loadout.Find("caitken-and-shollan");
		if (!caitkenAndShollanUsed && upgradeCardRuntimeState2 != null && upgradeCardRuntimeState2.Active && !upgradeCardRuntimeState2.Exhausted)
		{
			AttackDieColor[] array = new AttackDieColor[3]
			{
				AttackDieColor.Red,
				AttackDieColor.Blue,
				AttackDieColor.Black
			};
			for (int i = 0; i < array.Length; i++)
			{
				AttackDieColor attackDieColor = array[i];
				if (AttackResolver.HasDie(attack, attackDieColor))
				{
					int num = (int)attackDieColor;
					list.Add(new AbilityAction("attack.dice.caitken-and-shollan." + num, "caitken-and-shollan", upgradeCardRuntimeState2.Definition.name, AbilitySourceKind.Upgrade, "REROLL " + attackDieColor.ToString().ToUpperInvariant(), "Exhaust Caitken and Shollan, then choose any number of " + attackDieColor.ToString() + " dice to reroll.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState2.Definition.ability), AbilityTiming.ModifyAttackDice));
				}
			}
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState3 = attacker.Loadout.Find("ordnance-experts");
		if (!ordnanceExpertsUsed && upgradeCardRuntimeState3 != null && upgradeCardRuntimeState3.Active && AttackResolver.HasDie(attack, AttackDieColor.Black))
		{
			list.Add(new AbilityAction("attack.dice.ordnance-experts", "ordnance-experts", upgradeCardRuntimeState3.Definition.name, AbilitySourceKind.Upgrade, "ORDNANCE EXPERTS: REROLL UP TO 2", "Choose one or two black attack dice to reroll.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState3.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		if (!screedUsed && attack.Dice.Count > 1 && CommanderRules.FleetHasCommander(attacker, "admiral-screed-commander"))
		{
			list.Add(new AbilityAction("attack.dice.admiral-screed", "admiral-screed-commander", "Admiral Screed", AbilitySourceKind.Upgrade, "ADMIRAL SCREED", "Choose one die to spend, then choose another die and change it to a face with a critical icon.", "Once per activation, spend one die to change another die to a face with a critical icon.", AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState4 = attacker.Loadout.Find("warlord-title");
		if (!warlordUsed && attack.Accuracies > 0 && upgradeCardRuntimeState4 != null && upgradeCardRuntimeState4.Active)
		{
			list.Add(new AbilityAction("attack.dice.warlord", "warlord-title", upgradeCardRuntimeState4.Definition.name, AbilitySourceKind.Upgrade, "WARLORD: ACCURACY TO HIT", "Change one accuracy result to the best face containing a hit icon.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState4.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState5 = attacker.Loadout.Find("h9-turbolasers");
		if (upgradeCardRuntimeState5 != null && upgradeCardRuntimeState5.Active && upgradeCardRuntimeState5.UsesThisAttack == 0 && attack.Dice.Any((AttackDie die) => die.Color != AttackDieColor.Black && (die.Face == DieFace.Hit || die.Face == DieFace.DoubleHit || die.Face == DieFace.Critical || die.Face == DieFace.HitCritical)))
		{
			list.Add(new AbilityAction("attack.dice.h9-turbolasers", "h9-turbolasers", upgradeCardRuntimeState5.Definition.name, AbilitySourceKind.Upgrade, "H9: SET ACCURACY", "Choose one die showing a hit or critical icon and change it to an accuracy result.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState5.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		if (!homeOneUsed && TitleRules.HomeOneSource(attacker) != null && attack.Dice.Any((AttackDie die) => die.Color != AttackDieColor.Black))
		{
			list.Add(new AbilityAction("attack.dice.home-one", "home-one-title", "Home One", AbilitySourceKind.Upgrade, "HOME ONE: SET ACCURACY", "Choose one die and change it to an accuracy result.", "While another friendly ship at distance 1-5 is attacking, it may change one die to a face with an accuracy icon.", AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState6 = attacker.Loadout.Find("veteran-gunners");
		if (upgradeCardRuntimeState6 != null && upgradeCardRuntimeState6.Active && !upgradeCardRuntimeState6.Exhausted)
		{
			list.Add(new AbilityAction("attack.dice.veteran-gunners", "veteran-gunners", upgradeCardRuntimeState6.Definition.name, AbilitySourceKind.Upgrade, "VETERAN GUNNERS: REROLL ALL", "Exhaust Veteran Gunners to reroll every die in the attack pool.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState6.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState7 = attacker.Loadout.Find("sensor-team");
		if (attack.Dice.Count > 1 && upgradeCardRuntimeState7 != null && upgradeCardRuntimeState7.Active && !upgradeCardRuntimeState7.Exhausted)
		{
			list.Add(new AbilityAction("attack.dice.sensor-team", "sensor-team", upgradeCardRuntimeState7.Definition.name, AbilitySourceKind.Upgrade, "SENSOR TEAM: SET ACCURACY", "Choose one die to spend, then choose another die to change to an accuracy result and exhaust Sensor Team.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState7.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState8 = attacker.Loadout.Find("weapons-battery-techs");
		if (attack.Accuracies > 0 && upgradeCardRuntimeState8 != null && upgradeCardRuntimeState8.Active && upgradeCardRuntimeState8.UsesThisAttack == 0)
		{
			AttackDieColor[] array = new AttackDieColor[3]
			{
				AttackDieColor.Red,
				AttackDieColor.Blue,
				AttackDieColor.Black
			};
			foreach (AttackDieColor attackDieColor2 in array)
			{
				if (AttackResolver.HasDie(attack, attackDieColor2, DieFace.Accuracy))
				{
					string weaponClass = AttackResolver.GetWeaponClass(attackDieColor2);
					int num = (int)attackDieColor2;
					list.Add(new AbilityAction("attack.dice.weapons-battery-techs." + num, "weapons-battery-techs", upgradeCardRuntimeState8.Definition.name, AbilitySourceKind.Upgrade, "BATTERY TECHS: " + weaponClass.ToUpperInvariant() + " ACCURACY TO CRIT", "Change one " + weaponClass.ToLowerInvariant() + " weapon accuracy result to its face containing a critical icon.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState8.Definition.ability), AbilityTiming.ModifyAttackDice));
				}
			}
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState9 = attacker.Loadout.Find("linked-turbolaser-towers");
		if (AttackResolver.HasDie(attack, AttackDieColor.Red) && upgradeCardRuntimeState9 != null && upgradeCardRuntimeState9.Active && (againstSquadron || HasBombardTrait(attacker)) && upgradeCardRuntimeState9.UsesThisAttack == 0)
		{
			list.Add(new AbilityAction("attack.dice.linked-turbolaser-towers", "linked-turbolaser-towers", upgradeCardRuntimeState9.Definition.name, AbilitySourceKind.Upgrade, "LINKED TURRETS: REROLL RED", "Choose one red die in this attack pool and reroll it.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState9.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState10 = attacker.Loadout.Find("dual-turbolaser-turrets");
		if (upgradeCardRuntimeState10 != null && upgradeCardRuntimeState10.Active && !upgradeCardRuntimeState10.Exhausted)
		{
			list.Add(new AbilityAction("attack.dice.dual-turbolaser-turrets", "dual-turbolaser-turrets", upgradeCardRuntimeState10.Definition.name, AbilitySourceKind.Upgrade, "DUAL TURRETS: CYCLE RED DIE", "Exhaust Dual Turbolaser Turrets to roll one additional red die, then choose one die to remove.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState10.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState11 = attacker.Loadout.Find("turbolaser-reroute-circuits");
		if (AttackResolver.HasDie(attack, AttackDieColor.Red) && upgradeCardRuntimeState11 != null && upgradeCardRuntimeState11.Active && !upgradeCardRuntimeState11.Exhausted && attacker.State.HasDefenseToken(DefenseTokenType.Evade))
		{
			list.Add(new AbilityAction("attack.dice.turbolaser-reroute-circuits.critical", "turbolaser-reroute-circuits", upgradeCardRuntimeState11.Definition.name, AbilitySourceKind.Upgrade, "TRC: SET RED CRITICAL", "Exhaust this card and spend an Evade token to change one red die to a critical result.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState11.Definition.ability), AbilityTiming.ModifyAttackDice));
			list.Add(new AbilityAction("attack.dice.turbolaser-reroute-circuits.double-hit", "turbolaser-reroute-circuits", upgradeCardRuntimeState11.Definition.name, AbilitySourceKind.Upgrade, "TRC: SET RED DOUBLE HIT", "Exhaust this card and spend an Evade token to change one red die to a double-hit result.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState11.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		return list;
	}

	public static IReadOnlyList<AbilityAction> GetConditionalAttackDiceActions(ShipView attacker, ShipView defender, RangeBand range, bool defenderActivatedThisRound, bool firstAttackThisActivation)
	{
		List<AbilityAction> list = new List<AbilityAction>();
		if (attacker?.Loadout == null || defender == null)
		{
			return list;
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState = attacker.Loadout.Find("defiance-title");
		if (defenderActivatedThisRound && upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && upgradeCardRuntimeState.UsesThisAttack == 0)
		{
			AttackDieColor[] array = new AttackDieColor[3]
			{
				AttackDieColor.Red,
				AttackDieColor.Blue,
				AttackDieColor.Black
			};
			for (int i = 0; i < array.Length; i++)
			{
				AttackDieColor attackDieColor = array[i];
				int num = (int)attackDieColor;
				list.Add(new AbilityAction("attack.dice.defiance." + num, "defiance-title", upgradeCardRuntimeState.Definition.name, AbilitySourceKind.Upgrade, "DEFIANCE: ADD " + attackDieColor.ToString().ToUpperInvariant(), "Add one " + attackDieColor.ToString() + " die because the defender already activated this round.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState.Definition.ability), AbilityTiming.ModifyAttackDice));
			}
		}
		int num2 = 0;
		foreach (HullZone hullZone in attacker.Definition.HullZones)
		{
			num2 += attacker.State.GetShields(hullZone);
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState2 = attacker.Loadout.Find("dominator-title");
		if ((range == RangeBand.Close || range == RangeBand.Medium) && upgradeCardRuntimeState2 != null && upgradeCardRuntimeState2.Active && upgradeCardRuntimeState2.UsesThisAttack == 0)
		{
			for (int j = 1; j <= Math.Min(2, num2); j++)
			{
				list.Add(new AbilityAction("attack.dice.dominator." + j, "dominator-title", upgradeCardRuntimeState2.Definition.name, AbilitySourceKind.Upgrade, "DOMINATOR: SPEND " + j + " SHIELD" + ((j == 1) ? string.Empty : "S"), "Spend " + j + " shield" + ((j == 1) ? string.Empty : "s") + " from hull zones you choose to add " + j + " blue " + ((j == 1) ? "die." : "dice."), RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState2.Definition.ability), AbilityTiming.ModifyAttackDice));
			}
		}
		UpgradeCardRuntimeState upgradeCardRuntimeState3 = attacker.Loadout.Find("patriot-fist-title");
		if (firstAttackThisActivation && (range == RangeBand.Medium || range == RangeBand.Long) && upgradeCardRuntimeState3 != null && upgradeCardRuntimeState3.Active && upgradeCardRuntimeState3.UsesThisActivation == 0)
		{
			list.Add(new AbilityAction("attack.dice.patriot-fist", "patriot-fist-title", upgradeCardRuntimeState3.Definition.name, AbilitySourceKind.Upgrade, "PATRIOT FIST: +2 BLUE", "Add two blue dice to this first attack. This ship cannot attack again during this activation.", RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState3.Definition.ability), AbilityTiming.ModifyAttackDice));
		}
		return list;
	}

	public static bool PatriotFistLimitsFurtherAttacks(ShipView ship)
	{
		if ((object)ship == null)
		{
			return false;
		}
		return ship.Loadout?.Find("patriot-fist-title")?.UsesThisActivation > 0;
	}

	private static void AddDamageReductionAction(ICollection<AbilityAction> actions, ShipView defender, string cardId, string abilityId, string label, string prompt)
	{
		UpgradeCardRuntimeState upgradeCardRuntimeState = defender.Loadout?.Find(cardId);
		if (upgradeCardRuntimeState != null && upgradeCardRuntimeState.Active && !upgradeCardRuntimeState.Exhausted)
		{
			actions.Add(new AbilityAction(abilityId, cardId, upgradeCardRuntimeState.Definition.name, AbilitySourceKind.Upgrade, label, prompt, RulesTextPresenter.ToVideoGameTerms(upgradeCardRuntimeState.Definition.ability), AbilityTiming.Defense));
		}
	}

	private static void AddShipAccuracyAction(ICollection<AbilityAction> actions, ShipView defender, DefenseTokenType token, string abilityId, string label, ISet<DefenseTokenType> lockedTokens)
	{
		if ((lockedTokens == null || !lockedTokens.Contains(token)) && DefenseResolver.CanTargetWithAccuracy(defender.State, token))
		{
			actions.Add(new AbilityAction(abilityId, "accuracy-result", "Accuracy result", AbilitySourceKind.Chassis, label, $"Spend one accuracy result to lock {token}.", $"The defending ship cannot spend its {token} defense token " + "against this attack.", AbilityTiming.ResolveAttackEffects));
		}
	}

	private static void AddShipDefenseAction(ICollection<AbilityAction> actions, ShipView defender, DefenseTokenType token, string abilityId, string label, string rulesText, ISet<DefenseTokenType> lockedTokens, ISet<DefenseTokenType> spentThisAttack)
	{
		if ((lockedTokens != null && lockedTokens.Contains(token)) || (spentThisAttack != null && spentThisAttack.Contains(token)) || !defender.State.CanResolveDefenseToken(token))
		{
			return;
		}
		int num;
		string text;
		if (defender.State.GetReadyDefenseTokenCount(token) == 0)
		{
			num = ((defender.State.GetAvailableDefenseTokenCount(token) > 0) ? 1 : 0);
			if (num != 0 && label.StartsWith("SPEND ", StringComparison.Ordinal))
			{
				text = "DISCARD " + label.Substring("SPEND ".Length);
				goto IL_007c;
			}
		}
		else
		{
			num = 0;
		}
		text = label;
		goto IL_007c;
		IL_007c:
		string label2 = text;
		string text2 = ((num != 0) ? (rulesText + " This is your exhausted copy, so using it discards the token.") : rulesText);
		actions.Add(new AbilityAction(abilityId, token.ToString().ToLowerInvariant(), $"{token} defense token", AbilitySourceKind.Chassis, label2, text2, text2, AbilityTiming.Defense));
	}
}
