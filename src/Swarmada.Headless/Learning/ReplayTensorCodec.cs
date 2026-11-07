using System.IO.Compression;
using System.Text;

namespace Swarmada.Headless.Learning;

/// <summary>Lossless, versioned binary model-input transport. No snapshots,
/// authoritative secrets, padding or JSON float expansions are serialized.
/// Catalog one-hots and geometry are retained exactly; compression is separate
/// from numeric precision. This initial codec deliberately favors correctness
/// over a future catalog-manifest/delta representation.</summary>
public static class ReplayTensorCodec
{
    private const uint Magic = 0x31454D41; // AME1
    public static byte[] Encode(TensorRecord record, bool compress = true)
    {
        using var raw = new MemoryStream();
        using (var writer = new BinaryWriter(raw, Encoding.UTF8, true))
        {
            writer.Write(Magic); writer.Write(record.SchemaVersion); writer.Write(record.Tensors.Count);
            foreach (var tensor in record.Tensors)
            {
                tensor.Validate(); writer.Write(tensor.Name); writer.Write(tensor.Shape.Length);
                foreach (int dimension in tensor.Shape) writer.Write(dimension);
                writer.Write(tensor.DType); writer.Write(tensor.Data.Length);
                byte[] bytes = new byte[checked((int)tensor.ByteSize)]; Buffer.BlockCopy(tensor.Data, 0, bytes, 0, bytes.Length); writer.Write(bytes);
            }
        }
        using var output = new MemoryStream(); output.WriteByte(compress ? (byte)2 : (byte)0);
        if (compress)
        {
            using (var compressor = new ZLibStream(output, CompressionLevel.Fastest, true))
                compressor.Write(raw.GetBuffer().AsSpan(0, checked((int)raw.Length)));
        }
        else output.Write(raw.GetBuffer().AsSpan(0, checked((int)raw.Length)));
        return output.ToArray();
    }

    public static TensorRecord Decode(byte[] bytes, int maximumDecodedBytes = 64 * 1024 * 1024)
    {
        if (bytes.Length == 0) throw new InvalidDataException("Empty encoder record.");
        using var source = new MemoryStream(bytes, 1, bytes.Length - 1, false);
        using var raw = new MemoryStream();
        if (bytes[0] is 1 or 2)
        {
            using Stream decompressor = bytes[0] == 1 ? new BrotliStream(source, CompressionMode.Decompress, true)
                : new ZLibStream(source, CompressionMode.Decompress, true);
            var buffer = new byte[8192]; int count;
            while ((count = decompressor.Read(buffer)) != 0)
            {
                if (raw.Length + count > maximumDecodedBytes) throw new InvalidDataException("Encoder record exceeds its configured memory limit.");
                raw.Write(buffer, 0, count);
            }
        }
        else if (bytes[0] == 0)
        {
            if (source.Length > maximumDecodedBytes) throw new InvalidDataException("Encoder record exceeds its configured memory limit.");
            source.CopyTo(raw);
        }
        else throw new InvalidDataException("Unknown encoder transport compression.");
        raw.Position = 0; using var reader = new BinaryReader(raw, Encoding.UTF8, true);
        if (reader.ReadUInt32() != Magic) throw new InvalidDataException("Encoder transport version mismatch.");
        string schema = reader.ReadString(); int tensors = reader.ReadInt32();
        if (tensors < 0 || tensors > 256) throw new InvalidDataException("Invalid tensor registry size.");
        var result = new List<EncodedTensor>(tensors); var names = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < tensors; i++)
        {
            string name = reader.ReadString(); int rank = reader.ReadInt32();
            if (!names.Add(name) || rank < 0 || rank > 8) throw new InvalidDataException("Invalid tensor name/rank.");
            var shape = new int[rank]; long elements = 1;
            for (int axis = 0; axis < rank; axis++)
            {
                int dimension = reader.ReadInt32();
                if (dimension < 0) throw new InvalidDataException("Negative tensor dimension.");
                shape[axis] = dimension; elements = checked(elements * dimension);
            }
            string type = reader.ReadString(); int length = reader.ReadInt32();
            int elementBytes = type is "float32" or "int32" ? 4 : type is "uint8" or "bool" ? 1 : throw new InvalidDataException("Unknown tensor dtype.");
            if (length != elements || elements > maximumDecodedBytes / elementBytes) throw new InvalidDataException("Invalid tensor length.");
            int byteCount = checked(length * elementBytes);
            if (raw.Length - raw.Position < byteCount) throw new InvalidDataException("Truncated tensor payload.");
            Array data = type switch { "float32" => new float[length], "int32" => new int[length], "uint8" => new byte[length], _ => new bool[length] };
            byte[] contents = reader.ReadBytes(byteCount); Buffer.BlockCopy(contents, 0, data, 0, byteCount);
            var tensor = new EncodedTensor(name, shape, data); tensor.Validate(); result.Add(tensor);
        }
        if (raw.Position != raw.Length) throw new InvalidDataException("Trailing data after the encoder tensor registry.");
        return new(schema, result);
    }
}
