using System.Numerics;

namespace Swarmada.Headless.Geometry;

/// <summary>Immutable local-space query node. Leaves reference original edges or
/// triangles; the index changes traversal cost without changing the outline.</summary>
public readonly record struct PolygonQueryNode(Vector2 Minimum, Vector2 Maximum, int Left, int Right, int First, int Count)
{
    public bool IsLeaf => Count != 0;
}

/// <summary>Immutable physical outline. Concave contours are triangulated once, outside
/// search branches. Coordinates retain the asset's physical pivot and millimetre scale.</summary>
public sealed class PolygonDefinition
{
    private readonly Vector2[] vertices;
    private readonly int[] triangles;
    private readonly int[][] convexCells;
    private readonly PolygonQueryNode[] edgeNodes, triangleNodes;
    private readonly int[] edgeOrder, triangleOrder;
    public string Id { get; }
    public string Provenance { get; }
    public Vector2 Size { get; }
    public Vector2 BoundsMinimum { get; }
    public Vector2 BoundsMaximum { get; }
    public Circle InteriorCircle { get; }
    public int Count => vertices.Length;
    public int TriangleCount => triangles.Length / 3;
    public int ConvexCellCount => convexCells.Length;
    public int MaximumConvexCellVertices { get; }
    public ReadOnlySpan<Vector2> Vertices => vertices;
    public PolygonDefinition(string id, Vector2 size, ReadOnlySpan<Vector2> outline, string provenance = "")
    {
        if (outline.Length < 3) throw new ArgumentException("An outline requires at least three vertices.", nameof(outline));
        Geometry2D.RequireFinite(size, nameof(size)); if (size.X <= 0 || size.Y <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        Id = id; Size = size; Provenance = provenance; vertices = outline.ToArray();
        var minimum = new Vector2(float.PositiveInfinity); var maximum = new Vector2(float.NegativeInfinity); double area = 0;
        for (int i = 0; i < vertices.Length; i++)
        {
            Geometry2D.RequireFinite(vertices[i], nameof(outline)); minimum = Vector2.Min(minimum, vertices[i]); maximum = Vector2.Max(maximum, vertices[i]);
            if (Vector2.DistanceSquared(vertices[i], vertices[(i + 1) % vertices.Length]) < 1e-12f) throw new ArgumentException("Adjacent outline vertices must differ.", nameof(outline));
            area += Geometry2D.Cross(vertices[i], vertices[(i + 1) % vertices.Length]);
        }
        if (Math.Abs(area) < 1e-8) throw new ArgumentException("An outline must have nonzero area.", nameof(outline));
        if (area < 0) Array.Reverse(vertices);
        BoundsMinimum = minimum; BoundsMaximum = maximum; triangles = Triangulate(vertices);
        convexCells = BuildConvexCells(); MaximumConvexCellVertices = convexCells.Max(cell => cell.Length);
        (edgeNodes, edgeOrder) = BuildIndex(vertices.Length, primitive =>
        {
            var first = vertices[primitive]; var second = vertices[(primitive + 1) % vertices.Length];
            return (Vector2.Min(first, second), Vector2.Max(first, second));
        });
        (triangleNodes, triangleOrder) = BuildIndex(TriangleCount, primitive =>
        {
            var a = vertices[triangles[primitive * 3]]; var b = vertices[triangles[primitive * 3 + 1]]; var c = vertices[triangles[primitive * 3 + 2]];
            return (Vector2.Min(a, Vector2.Min(b, c)), Vector2.Max(a, Vector2.Max(b, c)));
        });
        InteriorCircle = FindInteriorCircle();
    }
    internal Vector2 Vertex(int index) => vertices[index];
    internal int TriangleVertex(int triangle, int corner) => triangles[triangle * 3 + corner];
    internal ReadOnlySpan<int> ConvexCell(int cell) => convexCells[cell];
    private int[][] BuildConvexCells()
    {
        // Remove only internal triangulation diagonals. Every surviving edge is an
        // original triangle edge, and strict convexity checks never take a hull or
        // replace a concave boundary by a shortcut.
        var cells = new List<int[]>();
        for (int t = 0; t < TriangleCount; t++) cells.Add(triangles.AsSpan(t * 3, 3).ToArray());
        bool changed;
        do
        {
            changed = false;
            var edges = new Dictionary<(int First, int Second), (int Cell, int Edge)>();
            for (int c = 0; c < cells.Count && !changed; c++)
            {
                var current = cells[c];
                for (int e = 0; e < current.Length; e++)
                {
                    int first = current[e], second = current[(e + 1) % current.Length];
                    if (edges.TryGetValue((second, first), out var neighbor))
                    {
                        var other = cells[neighbor.Cell];
                        if (current.Count(vertex => Array.IndexOf(other, vertex) >= 0) != 2) continue;
                        var merged = new int[current.Length + other.Length - 2];
                        for (int k = 0; k < current.Length; k++) merged[k] = current[(e + 1 + k) % current.Length];
                        for (int k = 1; k < other.Length - 1; k++) merged[current.Length + k - 1] = other[(neighbor.Edge + 1 + k) % other.Length];
                        bool convex = true;
                        for (int k = 0; k < merged.Length; k++)
                        {
                            var a = vertices[merged[k]]; var b = vertices[merged[(k + 1) % merged.Length]]; var d = vertices[merged[(k + 2) % merged.Length]];
                            if (Geometry2D.Cross(b - a, d - b) < 0) { convex = false; break; }
                        }
                        if (!convex) continue;
                        cells[neighbor.Cell] = merged; cells.RemoveAt(c); changed = true; break;
                    }
                    edges[(first, second)] = (c, e);
                }
            }
        } while (changed);
        return cells.ToArray();
    }
    public PolygonQueryNode EdgeNode(int node) => edgeNodes[node];
    public PolygonQueryNode TriangleNode(int node) => triangleNodes[node];
    public int EdgeIndex(int leafIndex) => edgeOrder[leafIndex];
    public int TriangleIndex(int leafIndex) => triangleOrder[leafIndex];
    private Circle FindInteriorCircle()
    {
        var shape = new PolygonFootprint(this, new Pose(Vector2.Zero), Size);
        var bestPoint = vertices[triangles[0]]; float radius = 0;
        void Consider(Vector2 point)
        {
            if (!shape.Contains(point, 0)) return;
            float candidate = shape.DistanceToBoundary(point);
            if (candidate > radius) { radius = candidate; bestPoint = point; }
        }
        Consider(Vector2.Zero); Consider((BoundsMinimum + BoundsMaximum) / 2);
        double twiceArea = 0, x = 0, y = 0;
        for (int i = 0; i < vertices.Length; i++)
        {
            var a = vertices[i]; var b = vertices[(i + 1) % vertices.Length]; double cross = Geometry2D.Cross(a, b);
            twiceArea += cross; x += (a.X + b.X) * cross; y += (a.Y + b.Y) * cross;
        }
        Consider(new((float)(x / (3 * twiceArea)), (float)(y / (3 * twiceArea))));
        for (int i = 0; i < TriangleCount; i++)
        {
            var a = vertices[triangles[i * 3]]; var b = vertices[triangles[i * 3 + 1]]; var c = vertices[triangles[i * 3 + 2]];
            float wa = Vector2.Distance(b, c), wb = Vector2.Distance(a, c), wc = Vector2.Distance(a, b);
            Consider((wa * a + wb * b + wc * c) / (wa + wb + wc));
        }
        // Round inward so this remains a certificate, never a visual approximation.
        return new(bestPoint, MathF.Max(float.Epsilon, radius - MathF.Min(Geometry2D.DefaultTolerance, radius / 2)));
    }
    private static (PolygonQueryNode[], int[]) BuildIndex(int count, Func<int, (Vector2 Minimum, Vector2 Maximum)> bounds)
    {
        var order = Enumerable.Range(0, count).ToArray(); var nodes = new List<PolygonQueryNode>(count);
        var minimums = new Vector2[count]; var maximums = new Vector2[count];
        for (int i = 0; i < count; i++) (minimums[i], maximums[i]) = bounds(i);
        int Build(int first, int length)
        {
            int index = nodes.Count; nodes.Add(default);
            var minimum = new Vector2(float.PositiveInfinity); var maximum = new Vector2(float.NegativeInfinity);
            for (int i = first; i < first + length; i++) { minimum = Vector2.Min(minimum, minimums[order[i]]); maximum = Vector2.Max(maximum, maximums[order[i]]); }
            if (length <= 4) { nodes[index] = new(minimum, maximum, -1, -1, first, length); return index; }
            bool x = maximum.X - minimum.X >= maximum.Y - minimum.Y;
            Array.Sort(order, first, length, Comparer<int>.Create((a, b) =>
                (x ? minimums[a].X + maximums[a].X : minimums[a].Y + maximums[a].Y).CompareTo(
                    x ? minimums[b].X + maximums[b].X : minimums[b].Y + maximums[b].Y)));
            int half = length / 2; int left = Build(first, half), right = Build(first + half, length - half);
            nodes[index] = new(minimum, maximum, left, right, 0, 0); return index;
        }
        Build(0, count); return (nodes.ToArray(), order);
    }
    private static int[] Triangulate(Vector2[] points)
    {
        var remaining = Enumerable.Range(0, points.Length).ToList(); var result = new List<int>((points.Length - 2) * 3);
        while (remaining.Count > 3)
        {
            bool removed = false;
            for (int i = 0; i < remaining.Count; i++)
            {
                int a = remaining[(i + remaining.Count - 1) % remaining.Count], b = remaining[i], c = remaining[(i + 1) % remaining.Count];
                float cross = Geometry2D.Cross(points[b] - points[a], points[c] - points[b]);
                if (MathF.Abs(cross) <= 1e-8f) { remaining.RemoveAt(i); removed = true; break; }
                if (cross < 0) continue;
                bool contains = false;
                foreach (int other in remaining)
                    if (other != a && other != b && other != c && InTriangle(points[other], points[a], points[b], points[c])) { contains = true; break; }
                if (contains) continue;
                result.Add(a); result.Add(b); result.Add(c); remaining.RemoveAt(i); removed = true; break;
            }
            if (!removed) throw new ArgumentException("Outline must be a simple polygon without holes or self-intersection.");
        }
        if (remaining.Count == 3) result.AddRange(remaining);
        return result.ToArray();
    }
    private static bool InTriangle(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        => Geometry2D.Cross(b - a, p - a) >= -1e-8f && Geometry2D.Cross(c - b, p - b) >= -1e-8f && Geometry2D.Cross(a - c, p - c) >= -1e-8f;
}

/// <summary>Allocation-free placed view of a shared physical outline. Flipping reflects
/// local X before rotation; pose, size and flip are all branch-owned spatial state.</summary>
public readonly struct PolygonFootprint
{
    private readonly Vector2 xAxis, yAxis;
    public PolygonDefinition Definition { get; }
    public Pose Pose { get; }
    public Vector2 Size { get; }
    public bool Flipped { get; }
    public Vector2 Center => Pose.Position;
    public int Count => Definition.Count;
    public int TriangleCount => Definition.TriangleCount;
    public int ConvexCellCount => Definition.ConvexCellCount;
    public Circle InteriorCircle
    {
        get
        {
            var local = Definition.InteriorCircle;
            return new(Pose.Position + xAxis * local.Center.X + yAxis * local.Center.Y,
                local.Radius * MathF.Min(xAxis.Length(), yAxis.Length()));
        }
    }
    public PolygonFootprint(PolygonDefinition definition, Pose pose, Vector2 size, bool flipped = false)
    {
        ArgumentNullException.ThrowIfNull(definition); Geometry2D.RequireFinite(size, nameof(size));
        if (size.X <= 0 || size.Y <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        Definition = definition; Pose = pose; Size = size; Flipped = flipped;
        xAxis = pose.Right * (size.X / definition.Size.X * (flipped ? -1 : 1)); yAxis = pose.Forward * (size.Y / definition.Size.Y);
    }
    public Vector2 Vertex(int index)
    { var local = Definition.Vertex(index); return Pose.Position + xAxis * local.X + yAxis * local.Y; }
    public Segment Edge(int index) => new(Vertex(index), Vertex((index + 1) % Count));
    public void CopyVertices(Span<Vector2> destination)
    { if (destination.Length < Count) throw new ArgumentException("Insufficient vertex storage.", nameof(destination)); for (int i = 0; i < Count; i++) destination[i] = Vertex(i); }
    public void CopyTriangle(int triangle, Span<Vector2> destination)
    {
        if (destination.Length < 3) throw new ArgumentException("Three triangle vertices are required.", nameof(destination));
        destination[0] = Vertex(Definition.TriangleVertex(triangle, 0));
        destination[1] = Vertex(Definition.TriangleVertex(triangle, Flipped ? 2 : 1));
        destination[2] = Vertex(Definition.TriangleVertex(triangle, Flipped ? 1 : 2));
    }
    /// <summary>Copy one exact convex part with counterclockwise winding on either face.</summary>
    public int CopyConvexCell(int cell, Span<Vector2> destination)
    {
        var indices = Definition.ConvexCell(cell);
        if (destination.Length < indices.Length) throw new ArgumentException("Insufficient convex-cell storage.", nameof(destination));
        for (int i = 0; i < indices.Length; i++) destination[i] = Vertex(indices[Flipped ? indices.Length - 1 - i : i]);
        return indices.Length;
    }
    public PolygonFootprint At(Pose pose, bool? flipped = null) => new(Definition, pose, Size, flipped ?? Flipped);
    internal void NodeWorldBounds(PolygonQueryNode node, out Vector2 minimum, out Vector2 maximum)
    {
        var half = (node.Maximum - node.Minimum) / 2; var localCenter = (node.Maximum + node.Minimum) / 2;
        var center = Pose.Position + xAxis * localCenter.X + yAxis * localCenter.Y;
        var extent = Vector2.Abs(xAxis) * half.X + Vector2.Abs(yAxis) * half.Y;
        minimum = center - extent; maximum = center + extent;
    }
    public OrientedRectangle TriangleNodeBounds(int index)
    {
        var node = Definition.TriangleNode(index); var half = (node.Maximum - node.Minimum) / 2;
        var localCenter = (node.Maximum + node.Minimum) / 2;
        return new(new(Pose.Position + xAxis * localCenter.X + yAxis * localCenter.Y, Pose.YawDegrees),
            new(xAxis.Length() * half.X * 2, yAxis.Length() * half.Y * 2));
    }
    /// <summary>Conservative exact-query candidates, with caller-owned scratch storage.
    /// Only bounds prune candidates; callers still run their ordinary narrow check.</summary>
    public int QueryTriangles(OrientedRectangle worldBounds, Span<int> result, float tolerance = Geometry2D.DefaultTolerance)
    {
        Geometry2D.RequireTolerance(tolerance); int count = 0;
        QueryTriangleNode(this, 0, worldBounds, result, ref count, tolerance); return count;
    }
    private static void QueryTriangleNode(PolygonFootprint polygon, int index, OrientedRectangle bounds, Span<int> result, ref int count, float tolerance)
    {
        var node = polygon.Definition.TriangleNode(index);
        if (Geometry2D.Contact(bounds, polygon.TriangleNodeBounds(index), tolerance) == ContactKind.Separated) return;
        if (node.IsLeaf)
        {
            if (count + node.Count > result.Length) throw new ArgumentException("Triangle query scratch storage is too small.", nameof(result));
            for (int i = 0; i < node.Count; i++) result[count++] = polygon.Definition.TriangleIndex(node.First + i);
            return;
        }
        QueryTriangleNode(polygon, node.Left, bounds, result, ref count, tolerance);
        QueryTriangleNode(polygon, node.Right, bounds, result, ref count, tolerance);
    }
    public OrientedRectangle Bounds
    {
        get
        {
            var half = (Definition.BoundsMaximum - Definition.BoundsMinimum) / 2;
            var center = (Definition.BoundsMaximum + Definition.BoundsMinimum) / 2;
            return new(new(Pose.Position + xAxis * center.X + yAxis * center.Y, Pose.YawDegrees),
                new(MathF.Abs(xAxis.Length() * half.X * 2), yAxis.Length() * half.Y * 2));
        }
    }
    public float ProjectedRadius(Vector2 axis)
    { float radius = 0; for (int i = 0; i < Count; i++) radius = MathF.Max(radius, MathF.Abs(Vector2.Dot(Vertex(i) - Center, axis))); return radius; }
    public bool Contains(Vector2 point, float tolerance = Geometry2D.DefaultTolerance)
    {
        if (DistanceToBoundary(point) <= tolerance) return true;
        var delta = point - Pose.Position;
        var localPoint = new Vector2(Vector2.Dot(delta, xAxis) / xAxis.LengthSquared(), Vector2.Dot(delta, yAxis) / yAxis.LengthSquared());
        bool inside = false;
        for (int i = 0, j = Count - 1; i < Count; j = i++)
        {
            var a = Definition.Vertex(i); var b = Definition.Vertex(j);
            if ((a.Y > localPoint.Y) != (b.Y > localPoint.Y) && localPoint.X < (b.X - a.X) * (localPoint.Y - a.Y) / (b.Y - a.Y) + a.X) inside = !inside;
        }
        return inside;
    }
    public float DistanceToBoundary(Vector2 point) => Vector2.Distance(point, ClosestBoundaryPoint(point));
    public Vector2 ClosestBoundaryPoint(Vector2 point)
    {
        return Geometry2D.ClosestBoundaryPoint(this, point);
    }
    public Vector2 ClosestPoint(Vector2 point) => Contains(point, 0) ? point : ClosestBoundaryPoint(point);
}
