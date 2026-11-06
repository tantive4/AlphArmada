#nullable disable
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using Armada.Session;

namespace Armada.Fleet;

public static class FleetStarterPresetReconciler
{
	public static void Stamp(FleetListDto fleet, string presetId, int revision)
	{
		if (fleet == null)
		{
			throw new ArgumentNullException("fleet");
		}
		if (string.IsNullOrWhiteSpace(presetId))
		{
			throw new ArgumentException("A starter preset ID is required.", "presetId");
		}
		if (revision <= 0)
		{
			throw new ArgumentOutOfRangeException("revision");
		}
		fleet.starterPresetId = presetId.Trim();
		fleet.starterPresetRevision = revision;
		fleet.starterPresetFingerprint = ComputeFingerprint(fleet);
	}

	public static FleetStarterPresetReconcileResult Reconcile(FleetJsonRepository repository, IList<FleetListDto> liveFleets, FleetListDto desired)
	{
		if (repository == null)
		{
			throw new ArgumentNullException("repository");
		}
		if (liveFleets == null)
		{
			throw new ArgumentNullException("liveFleets");
		}
		if (desired == null)
		{
			throw new ArgumentNullException("desired");
		}
		if (string.IsNullOrWhiteSpace(desired.starterPresetId) || desired.starterPresetRevision <= 0 || !string.Equals(desired.starterPresetFingerprint, ComputeFingerprint(desired), StringComparison.Ordinal))
		{
			throw new InvalidOperationException("The desired starter preset stamp is invalid.");
		}
		int num = -1;
		bool flag = false;
		for (int i = 0; i < liveFleets.Count; i++)
		{
			FleetListDto fleetListDto = liveFleets[i];
			if (fleetListDto == null)
			{
				continue;
			}
			bool flag2 = string.Equals(fleetListDto.fleetId, desired.fleetId, StringComparison.OrdinalIgnoreCase);
			if (flag2)
			{
				if (num >= 0)
				{
					return FleetStarterPresetReconcileResult.Protected;
				}
				num = i;
			}
			if (!flag2 && string.Equals(fleetListDto.starterPresetId, desired.starterPresetId, StringComparison.Ordinal))
			{
				flag = true;
			}
		}
		if (flag)
		{
			return FleetStarterPresetReconcileResult.Protected;
		}
		if (num < 0)
		{
			FleetListDto fleetListDto2 = FleetDraft.Clone(desired);
			repository.Save(fleetListDto2);
			liveFleets.Add(fleetListDto2);
			return FleetStarterPresetReconcileResult.Seeded;
		}
		FleetListDto fleetListDto3 = liveFleets[num];
		if (!string.Equals(fleetListDto3.starterPresetId, desired.starterPresetId, StringComparison.Ordinal) || fleetListDto3.starterPresetRevision <= 0 || !string.Equals(fleetListDto3.starterPresetFingerprint, ComputeFingerprint(fleetListDto3), StringComparison.Ordinal))
		{
			return FleetStarterPresetReconcileResult.Protected;
		}
		if (fleetListDto3.starterPresetRevision >= desired.starterPresetRevision)
		{
			return FleetStarterPresetReconcileResult.Unchanged;
		}
		FleetListDto fleetListDto4 = FleetDraft.Clone(desired);
		repository.Save(fleetListDto4);
		liveFleets[num] = fleetListDto4;
		return FleetStarterPresetReconcileResult.Updated;
	}

	public static string ComputeFingerprint(FleetListDto fleet)
	{
		if (fleet == null)
		{
			return string.Empty;
		}
		StringBuilder stringBuilder = new StringBuilder(1024);
		Append(stringBuilder, fleet.schemaVersion);
		Append(stringBuilder, fleet.fleetId);
		Append(stringBuilder, fleet.displayName);
		Append(stringBuilder, fleet.faction);
		Append(stringBuilder, fleet.pointLimit);
		Append(stringBuilder, fleet.rulesMode);
		ShipBuildDto[] array = fleet.ships ?? Array.Empty<ShipBuildDto>();
		Append(stringBuilder, array.Length);
		ShipBuildDto[] array2 = array;
		string[] array4;
		foreach (ShipBuildDto shipBuildDto in array2)
		{
			if (shipBuildDto == null)
			{
				Append(stringBuilder, "<null-ship>");
				continue;
			}
			Append(stringBuilder, shipBuildDto.schemaVersion);
			Append(stringBuilder, shipBuildDto.instanceId);
			Append(stringBuilder, shipBuildDto.shipDefinitionId);
			string[] array3 = shipBuildDto.upgradeIds ?? Array.Empty<string>();
			Append(stringBuilder, array3.Length);
			array4 = array3;
			foreach (string value in array4)
			{
				Append(stringBuilder, value);
			}
		}
		SquadronBuildDto[] array5 = fleet.squadrons ?? Array.Empty<SquadronBuildDto>();
		Append(stringBuilder, array5.Length);
		SquadronBuildDto[] array6 = array5;
		foreach (SquadronBuildDto squadronBuildDto in array6)
		{
			if (squadronBuildDto == null)
			{
				Append(stringBuilder, "<null-squadron>");
				continue;
			}
			Append(stringBuilder, squadronBuildDto.schemaVersion);
			Append(stringBuilder, squadronBuildDto.instanceId);
			Append(stringBuilder, squadronBuildDto.squadronDefinitionId);
			Append(stringBuilder, squadronBuildDto.leaderUpgradeId);
		}
		string[] array7 = fleet.objectiveIds ?? Array.Empty<string>();
		Append(stringBuilder, array7.Length);
		array4 = array7;
		foreach (string value2 in array4)
		{
			Append(stringBuilder, value2);
		}
		using SHA256 sHA = SHA256.Create();
		byte[] array8 = sHA.ComputeHash(Encoding.UTF8.GetBytes(stringBuilder.ToString()));
		StringBuilder stringBuilder2 = new StringBuilder(array8.Length * 2);
		byte[] array9 = array8;
		foreach (byte b in array9)
		{
			stringBuilder2.Append(b.ToString("x2"));
		}
		return stringBuilder2.ToString();
	}

	private static void Append(StringBuilder builder, int value)
	{
		builder.Append(value).Append('|');
	}

	private static void Append(StringBuilder builder, string value)
	{
		if (value == null)
		{
			builder.Append("-1:|");
		}
		else
		{
			builder.Append(value.Length).Append(':').Append(value)
				.Append('|');
		}
	}
}
