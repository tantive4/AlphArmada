using System.Numerics;

namespace Swarmada.Headless.Geometry;

public static partial class Geometry2D
{
    public static ContactKind Contact(Circle circle, PolygonFootprint polygon, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance); float penetration = circle.Radius - Vector2.Distance(circle.Center, polygon.ClosestPoint(circle.Center));
        return penetration < -tolerance ? ContactKind.Separated : penetration <= tolerance ? ContactKind.Touching : ContactKind.Overlapping;
    }
    public static ContactKind Contact(PolygonFootprint polygon, Circle circle, float tolerance = DefaultTolerance) => Contact(circle, polygon, tolerance);
    public static ContactKind Contact(OrientedRectangle rectangle, PolygonFootprint polygon, float tolerance = DefaultTolerance)
    {
        if (Contact(rectangle, polygon.Bounds, tolerance) == ContactKind.Separated) return ContactKind.Separated;
        return ContactRectangleNode(rectangle, polygon, 0, tolerance);
    }
    public static ContactKind Contact(PolygonFootprint polygon, OrientedRectangle rectangle, float tolerance = DefaultTolerance) => Contact(rectangle, polygon, tolerance);
    public static ContactKind Contact(PolygonFootprint first, PolygonFootprint second, float tolerance = DefaultTolerance)
    {
        if (Contact(first.Bounds, second.Bounds, tolerance) == ContactKind.Separated) return ContactKind.Separated;
        return ContactNodes(first, 0, second, 0, tolerance);
    }
    public static bool Overlaps(Circle circle, PolygonFootprint polygon, float tolerance = DefaultTolerance) => Contact(circle, polygon, tolerance) == ContactKind.Overlapping;
    public static bool Overlaps(PolygonFootprint polygon, Circle circle, float tolerance = DefaultTolerance) => Overlaps(circle, polygon, tolerance);
    public static bool Overlaps(OrientedRectangle rectangle, PolygonFootprint polygon, float tolerance = DefaultTolerance) => Contact(rectangle, polygon, tolerance) == ContactKind.Overlapping;
    public static bool Overlaps(PolygonFootprint polygon, OrientedRectangle rectangle, float tolerance = DefaultTolerance) => Overlaps(rectangle, polygon, tolerance);
    public static bool Overlaps(PolygonFootprint first, PolygonFootprint second, float tolerance = DefaultTolerance) => Contact(first, second, tolerance) == ContactKind.Overlapping;
    public static float Distance(Circle circle, PolygonFootprint polygon) => MathF.Max(0, Vector2.Distance(circle.Center, polygon.ClosestPoint(circle.Center)) - circle.Radius);
    public static float Distance(PolygonFootprint polygon, Circle circle) => Distance(circle, polygon);
    public static float Distance(OrientedRectangle rectangle, PolygonFootprint polygon) => ClosestPoints(rectangle, polygon).Distance;
    public static float Distance(PolygonFootprint polygon, OrientedRectangle rectangle) => ClosestPoints(polygon, rectangle).Distance;
    public static float Distance(PolygonFootprint first, PolygonFootprint second) => ClosestPoints(first, second).Distance;
    public static ClosestPair ClosestPoints(OrientedRectangle rectangle, PolygonFootprint polygon)
    {
        Span<Vector2> corners = stackalloc Vector2[4]; rectangle.CopyCorners(corners);
        foreach (var point in corners) if (polygon.Contains(point, 0)) return new(0, point, point);
        var vertex = polygon.Vertex(0); if (rectangle.Contains(vertex, 0)) return new(0, vertex, vertex);
        var best = new ClosestPair(float.PositiveInfinity, corners[0], polygon.Vertex(0));
        ClosestRectangleNode(rectangle, polygon, 0, ref best);
        return best;
    }
    public static ClosestPair ClosestPoints(PolygonFootprint polygon, OrientedRectangle rectangle)
    { var pair = ClosestPoints(rectangle, polygon); return new(pair.Distance, pair.Second, pair.First); }
    public static ClosestPair ClosestPoints(PolygonFootprint first, PolygonFootprint second)
    {
        var firstPoint = first.Vertex(0); if (second.Contains(firstPoint, 0)) return new(0, firstPoint, firstPoint);
        var secondPoint = second.Vertex(0); if (first.Contains(secondPoint, 0)) return new(0, secondPoint, secondPoint);
        var best = new ClosestPair(float.PositiveInfinity, first.Vertex(0), second.Vertex(0));
        ClosestNodes(first, 0, second, 0, ref best);
        return best;
    }
    public static ClosestPair ClosestPoints(PolygonFootprint polygon, Circle circle)
    {
        var first = polygon.ClosestPoint(circle.Center); var delta = first - circle.Center; float distance = delta.Length();
        if (distance <= circle.Radius) return new(0, first, first);
        return new(distance - circle.Radius, first, circle.Center + delta * (circle.Radius / distance));
    }
    public static ClosestPair ClosestPoints(Circle circle, PolygonFootprint polygon)
    { var pair = ClosestPoints(polygon, circle); return new(pair.Distance, pair.Second, pair.First); }
    public static bool Contains(PolygonFootprint polygon, Circle circle, float tolerance = DefaultTolerance)
        => polygon.Contains(circle.Center, 0) && polygon.DistanceToBoundary(circle.Center) + tolerance >= circle.Radius;

    private static void RectangleBounds(OrientedRectangle rectangle, out Vector2 minimum, out Vector2 maximum)
    {
        var extent = new Vector2(rectangle.ProjectedRadius(Vector2.UnitX), rectangle.ProjectedRadius(Vector2.UnitY));
        minimum = rectangle.Center - extent; maximum = rectangle.Center + extent;
    }
    private static bool BoxesSeparated(Vector2 aMin, Vector2 aMax, Vector2 bMin, Vector2 bMax, float tolerance)
        => aMax.X < bMin.X - tolerance || bMax.X < aMin.X - tolerance || aMax.Y < bMin.Y - tolerance || bMax.Y < aMin.Y - tolerance;
    private static float BoxDistanceSquared(Vector2 aMin, Vector2 aMax, Vector2 bMin, Vector2 bMax)
    {
        var gap = Vector2.Max(Vector2.Zero, Vector2.Max(aMin - bMax, bMin - aMax)); return gap.LengthSquared();
    }
    private static ContactKind CombineContacts(ContactKind first, ContactKind second)
        => first == ContactKind.Overlapping || second == ContactKind.Overlapping ? ContactKind.Overlapping
            : first == ContactKind.Touching || second == ContactKind.Touching ? ContactKind.Touching : ContactKind.Separated;
    private static ContactKind ContactRectangleNode(OrientedRectangle rectangle, PolygonFootprint polygon, int index, float tolerance)
    {
        var node = polygon.Definition.TriangleNode(index); polygon.NodeWorldBounds(node, out var minimum, out var maximum);
        RectangleBounds(rectangle, out var rectangleMin, out var rectangleMax);
        if (BoxesSeparated(rectangleMin, rectangleMax, minimum, maximum, tolerance)) return ContactKind.Separated;
        if (!node.IsLeaf)
        {
            var first = ContactRectangleNode(rectangle, polygon, node.Left, tolerance);
            return first == ContactKind.Overlapping ? first : CombineContacts(first, ContactRectangleNode(rectangle, polygon, node.Right, tolerance));
        }
        Span<Vector2> corners = stackalloc Vector2[4]; Span<Vector2> triangle = stackalloc Vector2[3]; rectangle.CopyCorners(corners);
        var result = ContactKind.Separated;
        for (int i = 0; i < node.Count; i++)
        {
            polygon.CopyTriangle(polygon.Definition.TriangleIndex(node.First + i), triangle);
            var contact = Contact(corners, triangle, tolerance); if (contact == ContactKind.Overlapping) return contact;
            result = CombineContacts(result, contact);
        }
        return result;
    }
    private static ContactKind ContactNodes(PolygonFootprint first, int aIndex, PolygonFootprint second, int bIndex, float tolerance)
    {
        var a = first.Definition.TriangleNode(aIndex); var b = second.Definition.TriangleNode(bIndex);
        first.NodeWorldBounds(a, out var aMin, out var aMax); second.NodeWorldBounds(b, out var bMin, out var bMax);
        if (BoxesSeparated(aMin, aMax, bMin, bMax, tolerance)) return ContactKind.Separated;
        if (!a.IsLeaf || !b.IsLeaf)
        {
            bool splitA = !a.IsLeaf && (b.IsLeaf || (aMax - aMin).LengthSquared() >= (bMax - bMin).LengthSquared());
            var left = splitA ? ContactNodes(first, a.Left, second, bIndex, tolerance) : ContactNodes(first, aIndex, second, b.Left, tolerance);
            if (left == ContactKind.Overlapping) return left;
            return CombineContacts(left, splitA ? ContactNodes(first, a.Right, second, bIndex, tolerance) : ContactNodes(first, aIndex, second, b.Right, tolerance));
        }
        Span<Vector2> aTriangle = stackalloc Vector2[3]; Span<Vector2> bTriangle = stackalloc Vector2[3]; var result = ContactKind.Separated;
        for (int i = 0; i < a.Count; i++)
        {
            first.CopyTriangle(first.Definition.TriangleIndex(a.First + i), aTriangle);
            for (int j = 0; j < b.Count; j++)
            {
                second.CopyTriangle(second.Definition.TriangleIndex(b.First + j), bTriangle);
                var contact = Contact(aTriangle, bTriangle, tolerance); if (contact == ContactKind.Overlapping) return contact;
                result = CombineContacts(result, contact);
            }
        }
        return result;
    }
    internal static Vector2 ClosestBoundaryPoint(PolygonFootprint polygon, Vector2 point)
    {
        var best = polygon.Vertex(0); float squared = Vector2.DistanceSquared(point, best);
        ClosestPointNode(polygon, 0, point, ref best, ref squared); return best;
    }
    private static void ClosestPointNode(PolygonFootprint polygon, int index, Vector2 point, ref Vector2 best, ref float squared)
    {
        var node = polygon.Definition.EdgeNode(index); polygon.NodeWorldBounds(node, out var minimum, out var maximum);
        if (BoxDistanceSquared(point, point, minimum, maximum) > squared + 1e-6f) return;
        if (node.IsLeaf)
        {
            for (int i = 0; i < node.Count; i++)
            {
                var candidate = ClosestPoint(polygon.Edge(polygon.Definition.EdgeIndex(node.First + i)), point);
                float distance = Vector2.DistanceSquared(candidate, point); if (distance < squared) { squared = distance; best = candidate; }
            }
            return;
        }
        var left = polygon.Definition.EdgeNode(node.Left); var right = polygon.Definition.EdgeNode(node.Right);
        polygon.NodeWorldBounds(left, out var lMin, out var lMax); polygon.NodeWorldBounds(right, out var rMin, out var rMax);
        bool leftFirst = BoxDistanceSquared(point, point, lMin, lMax) <= BoxDistanceSquared(point, point, rMin, rMax);
        ClosestPointNode(polygon, leftFirst ? node.Left : node.Right, point, ref best, ref squared);
        ClosestPointNode(polygon, leftFirst ? node.Right : node.Left, point, ref best, ref squared);
    }
    private static void ClosestRectangleNode(OrientedRectangle rectangle, PolygonFootprint polygon, int index, ref ClosestPair best)
    {
        var node = polygon.Definition.EdgeNode(index); polygon.NodeWorldBounds(node, out var minimum, out var maximum);
        RectangleBounds(rectangle, out var rMin, out var rMax);
        if (BoxDistanceSquared(rMin, rMax, minimum, maximum) > best.Distance * best.Distance + 1e-6f) return;
        if (node.IsLeaf)
        {
            Span<Vector2> corners = stackalloc Vector2[4]; rectangle.CopyCorners(corners);
            for (int i = 0; i < node.Count; i++) for (int j = 0; j < 4; j++)
            {
                var pair = ClosestPoints(new Segment(corners[j], corners[(j + 1) % 4]), polygon.Edge(polygon.Definition.EdgeIndex(node.First + i)));
                if (pair.Distance < best.Distance) best = pair; if (best.Distance == 0) return;
            }
            return;
        }
        ClosestRectangleNode(rectangle, polygon, node.Left, ref best);
        if (best.Distance != 0) ClosestRectangleNode(rectangle, polygon, node.Right, ref best);
    }
    private static void ClosestNodes(PolygonFootprint first, int aIndex, PolygonFootprint second, int bIndex, ref ClosestPair best)
    {
        var a = first.Definition.EdgeNode(aIndex); var b = second.Definition.EdgeNode(bIndex);
        first.NodeWorldBounds(a, out var aMin, out var aMax); second.NodeWorldBounds(b, out var bMin, out var bMax);
        if (BoxDistanceSquared(aMin, aMax, bMin, bMax) > best.Distance * best.Distance + 1e-6f) return;
        if (a.IsLeaf && b.IsLeaf)
        {
            for (int i = 0; i < a.Count; i++) for (int j = 0; j < b.Count; j++)
            {
                var pair = ClosestPoints(first.Edge(first.Definition.EdgeIndex(a.First + i)), second.Edge(second.Definition.EdgeIndex(b.First + j)));
                if (pair.Distance < best.Distance) best = pair; if (best.Distance == 0) return;
            }
            return;
        }
        bool splitA = !a.IsLeaf && (b.IsLeaf || (aMax - aMin).LengthSquared() >= (bMax - bMin).LengthSquared());
        if (splitA) { ClosestNodes(first, a.Left, second, bIndex, ref best); if (best.Distance != 0) ClosestNodes(first, a.Right, second, bIndex, ref best); }
        else { ClosestNodes(first, aIndex, second, b.Left, ref best); if (best.Distance != 0) ClosestNodes(first, aIndex, second, b.Right, ref best); }
    }

    public static bool SegmentCrossesInterior(Segment segment, PolygonFootprint polygon, float tolerance = DefaultTolerance)
    {
        RequireTolerance(tolerance);
        if (segment.Length <= tolerance || !Clip(segment, polygon.Bounds, out _, out _)) return false;
        Span<Vector2> triangle = stackalloc Vector2[3];
        for (int i = 0; i < polygon.TriangleCount; i++)
        {
            polygon.CopyTriangle(i, triangle);
            if (!ClipConvex(segment, triangle, out float enter, out float exit) || (exit - enter) * segment.Length <= tolerance) continue;
            var midpoint = segment.PointAt((enter + exit) / 2);
            if (polygon.DistanceToBoundary(midpoint) > tolerance) return true;
        }
        return false;
    }
    private static bool ClipConvex(Segment segment, ReadOnlySpan<Vector2> vertices, out float enter, out float exit)
    {
        enter = 0; exit = 1;
        for (int i = 0; i < vertices.Length; i++)
        { var edge = vertices[(i + 1) % vertices.Length] - vertices[i]; if (!ClipPositive(Cross(edge, segment.Start - vertices[i]), Cross(edge, segment.Delta), ref enter, ref exit)) return false; }
        return true;
    }
}

public static partial class PlayAreaGeometry
{
    public static bool Contains(OrientedRectangle board, PolygonFootprint polygon, float tolerance = SourceContainmentTolerance)
    { for (int i = 0; i < polygon.Count; i++) if (!board.Contains(polygon.Vertex(i), tolerance)) return false; return true; }
}
