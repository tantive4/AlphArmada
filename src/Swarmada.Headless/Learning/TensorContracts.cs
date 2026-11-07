using System.Globalization;

namespace Swarmada.Headless.Learning;

/// <summary>Contiguous row-major, actual-count tensor. Integer values are categorical
/// IDs or exact references, not normalized numeric features. Padding is a batch concern.</summary>
public sealed record EncodedTensor(string Name, int[] Shape, Array Data)
{
    public string DType => Data switch
    {
        float[] => "float32", int[] => "int32", byte[] => "uint8", bool[] => "bool",
        _ => throw new ArgumentException("Unsupported encoded tensor element type.")
    };
    public long ByteSize => Buffer.ByteLength(Data);
    public EncodedTensor CloneOwned() => new(Name, (int[])Shape.Clone(), (Array)Data.Clone());
    public void Validate()
    {
        long elements = 1;
        foreach (int dimension in Shape)
        { if (dimension < 0) throw new ArgumentException("Negative tensor dimension."); elements = checked(elements * dimension); }
        if (elements != Data.LongLength || Data.Rank != 1) throw new ArgumentException("Tensor shape does not match its contiguous buffer: " + Name);
        _ = DType;
        if (Data is float[] values && values.Any(value => !float.IsFinite(value)))
            throw new ArgumentException("Nonfinite model feature: " + Name);
    }
    public string ShapeText => "[" + string.Join(",", Shape.Select(n => n.ToString(CultureInfo.InvariantCulture))) + "]";
}

public sealed record EncoderLimits(int MaximumShips = 24, int MaximumSquadrons = 48,
    int MaximumUpgrades = 384)
{
    internal void ValidateCounts(int ships, int squadrons, int upgrades)
    {
        if (ships > MaximumShips || squadrons > MaximumSquadrons || upgrades > MaximumUpgrades)
            throw new ArgumentException($"Fleet exceeds this encoder profile ({ships}/{MaximumShips} ships, " +
                $"{squadrons}/{MaximumSquadrons} squadrons, {upgrades}/{MaximumUpgrades} upgrades). " +
                "Supply a larger explicit profile; observations are never truncated.");
    }
}

public sealed record TensorRecord(string SchemaVersion, IReadOnlyList<EncodedTensor> Tensors)
{
    public long TensorBytes => Tensors.Sum(tensor => tensor.ByteSize);
}

/// <summary>Padding is allocated only for the current inference/training batch.
/// ActualShapes and row masks distinguish padding from destroyed/reserved units.</summary>
public sealed record EncodedBatch(IReadOnlyList<EncodedTensor> Tensors,
    IReadOnlyDictionary<string, int[][]> ActualShapes, int Samples)
{
    public long ByteSize => Tensors.Sum(tensor => tensor.ByteSize);
}

public static class EncoderBatchCollator
{
    public static EncodedBatch Collate(IReadOnlyList<EncodedSearchInput> inputs)
        => Collate(inputs.Select(input => input.ToRecord()).ToArray());

    public static EncodedBatch Collate(IReadOnlyList<TensorRecord> inputs)
    {
        if (inputs.Count == 0) throw new ArgumentException("Cannot collate an empty batch.", nameof(inputs));
        if (inputs.Any(input => input.SchemaVersion != inputs[0].SchemaVersion))
            throw new ArgumentException("Encoder schema/catalog fingerprints differ in one batch.");
        var lists = inputs.Select(input => input.Tensors.ToDictionary(t => t.Name, StringComparer.Ordinal)).ToArray();
        var result = new List<EncodedTensor>(); var actual = new Dictionary<string, int[][]>(StringComparer.Ordinal);
        foreach (string name in lists[0].Keys)
        {
            var sources = lists.Select(list => list.TryGetValue(name, out var value) ? value :
                throw new ArgumentException("Batch schemas differ: " + name)).ToArray();
            int rank = sources[0].Shape.Length;
            if (sources.Any(t => t.Shape.Length != rank || t.DType != sources[0].DType))
                throw new ArgumentException("Batch tensor types differ: " + name);
            int[] padded = Enumerable.Range(0, rank).Select(axis => sources.Max(t => t.Shape[axis])).ToArray();
            int stride = checked((int)padded.Aggregate(1L, (n, dimension) => n * dimension));
            Array buffer = Array.CreateInstance(sources[0].Data.GetType().GetElementType()!, checked(inputs.Count * stride));
            // Every reference tensor uses -1 for padding. Numeric features use zero.
            if (buffer is int[] references) Array.Fill(references, -1);
            for (int sample = 0; sample < sources.Length; sample++) CopyPadded(sources[sample], buffer, sample * stride, padded);
            result.Add(new(name, [inputs.Count, .. padded], buffer));
            actual[name] = sources.Select(source => (int[])source.Shape.Clone()).ToArray();
        }
        if (lists.Any(list => list.Count != lists[0].Count)) throw new ArgumentException("Batch tensor registries differ.");
        return new(result, actual, inputs.Count);
    }

    private static void CopyPadded(EncodedTensor source, Array destination, int sampleOffset, int[] padded)
    {
        if (source.Data.Length == 0) return;
        if (source.Shape.Length == 0) { Array.Copy(source.Data, 0, destination, sampleOffset, 1); return; }
        int last = source.Shape.Length - 1, width = source.Shape[last];
        int rows = source.Data.Length / width;
        for (int row = 0; row < rows; row++)
        {
            int remaining = row, destinationRow = 0, multiplier = 1;
            for (int axis = last - 1; axis >= 0; axis--)
            {
                int coordinate = remaining % source.Shape[axis]; remaining /= source.Shape[axis];
                destinationRow += coordinate * multiplier; multiplier *= padded[axis];
            }
            Array.Copy(source.Data, row * width, destination, sampleOffset + destinationRow * padded[last], width);
        }
    }
}
