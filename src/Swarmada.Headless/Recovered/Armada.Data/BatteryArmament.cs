using System;
using System.Text.Json;

namespace Armada.Data;

/// <summary>Recovered battery semantics with ordinary immutable values instead of Unity fields.</summary>
public readonly record struct BatteryArmament
{
    public int RedDice { get; }
    public int BlueDice { get; }
    public int BlackDice { get; }
    public int TotalDice => RedDice + BlueDice + BlackDice;

    public BatteryArmament(int red, int blue, int black)
    {
        RedDice = Math.Max(0, red);
        BlueDice = Math.Max(0, blue);
        BlackDice = Math.Max(0, black);
    }

    internal static BatteryArmament Read(JsonElement value) => new(
        value.GetProperty("redDice").GetInt32(), value.GetProperty("blueDice").GetInt32(),
        value.GetProperty("blackDice").GetInt32());
}
