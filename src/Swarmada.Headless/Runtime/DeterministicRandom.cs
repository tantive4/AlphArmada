using System.Numerics;

namespace Swarmada.Headless.Runtime;

public readonly record struct RandomSnapshot(ulong S0, ulong S1, ulong S2, ulong S3);

/// <summary>Explicit xoshiro256** state; independent of Unity and the runtime's System.Random version.</summary>
public struct DeterministicRandom
{
    private ulong s0, s1, s2, s3;
    public DeterministicRandom(ulong seed)
    {
        s0 = SplitMix(ref seed); s1 = SplitMix(ref seed); s2 = SplitMix(ref seed); s3 = SplitMix(ref seed);
    }
    public readonly RandomSnapshot CaptureSnapshot() => new(s0, s1, s2, s3);
    public void RestoreSnapshot(RandomSnapshot snapshot)
    {
        if ((snapshot.S0 | snapshot.S1 | snapshot.S2 | snapshot.S3) == 0)
            throw new ArgumentException("An all-zero random state is invalid.", nameof(snapshot));
        (s0, s1, s2, s3) = (snapshot.S0, snapshot.S1, snapshot.S2, snapshot.S3);
    }
    public ulong NextUInt64()
    {
        if ((s0 | s1 | s2 | s3) == 0)
            throw new InvalidOperationException("Initialize the random generator with a seed before sampling.");
        ulong result = BitOperations.RotateLeft(s1 * 5, 7) * 9;
        ulong t = s1 << 17;
        s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t;
        s3 = BitOperations.RotateLeft(s3, 45);
        return result;
    }
    public int NextInt(int exclusiveUpperBound)
    {
        if (exclusiveUpperBound < 1) throw new ArgumentOutOfRangeException(nameof(exclusiveUpperBound));
        ulong bound = (ulong)exclusiveUpperBound, threshold = unchecked(0UL - bound) % bound;
        ulong value;
        do { value = NextUInt64(); } while (value < threshold);
        return (int)(value % bound);
    }
    private static ulong SplitMix(ref ulong seed)
    {
        ulong z = seed += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}
