using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace Armada.Data;

/// <summary>Joint yaw is copied on construction; callers cannot mutate a shared maneuver chart.</summary>
public readonly struct ManeuverSpeedRow
{
    private readonly ReadOnlyCollection<int>? jointYaw;
    public IReadOnlyList<int> JointYaw => jointYaw ?? (IReadOnlyList<int>)Array.Empty<int>();
    public int JointCount => JointYaw.Count;
    public bool IsAvailable => JointCount > 0;

    public ManeuverSpeedRow(int[]? yawPerJoint) =>
        jointYaw = Array.AsReadOnly((int[])(yawPerJoint?.Clone() ?? Array.Empty<int>()));

    public int GetYaw(int joint) => joint < 0 || joint >= JointCount ? 0 : Math.Max(0, JointYaw[joint]);
}
