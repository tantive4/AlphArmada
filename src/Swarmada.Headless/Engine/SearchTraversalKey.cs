using System.Collections.Concurrent;
using System.Linq.Expressions;
using System.Reflection;
using Armada.Core;
using Armada.Rules;
using Swarmada.Headless.Engine;
using Swarmada.Headless.Runtime;

namespace Swarmada.Headless.Search;

/// <summary>Internal rule/knowledge discriminator, never a neural feature or a
/// persisted ID. It covers the mutable rules graph and root/acting knowledge;
/// inactive opponent private card identities use knowledge patterns until that
/// opponent chooses. This remains an approximate information-set tree.</summary>
public readonly record struct SearchTraversalKey(ulong Rules, ulong Knowledge);

/// <summary>Hashes storage directly, without snapshots, JSON or feature tensors.
/// Field readers are compiled once per runtime type. New rule-state fields are
/// included automatically. Unknown future deck order and RNG are deliberately
/// excluded: outcomes enter the key only once they affect the settled state.
/// This is a local tree discriminator, not a general transposition table.</summary>
internal static class SearchRuleKey
{
    public static SearchTraversalKey Create(GameSession game, IReadOnlyList<SearchKnowledge> knowledge,
        int observer, IReadOnlyList<string> prefix, int actingObserver = 0)
    {
        ulong rules = Add(game.Context.State, SearchHash.Initial);
        rules = Add(game.Context.Flow, rules);
        ulong information = SearchHash.Add(SearchHash.Initial, (ulong)observer);
        foreach (var view in knowledge)
            information = view.Observer == observer || view.Observer == actingObserver
                ? Add(view, information) : view.AddInactivePrivatePatternHash(information);
        for (int i = 0; i < prefix.Count; i++) information = SearchHash.Add(information, prefix[i]);
        return new(rules, information);
    }

    private static ulong Add<T>(T value, ulong hash) => Reader<T>.Hash(value, hash);
    private static ulong AddSequence<T>(IEnumerable<T> values, ulong hash)
    {
        int count = 0;
        foreach (T value in values) { hash = Add(value, hash); count++; }
        return SearchHash.Add(hash, (ulong)count);
    }
    private static ulong AddDictionary<TKey, TValue>(IDictionary<TKey, TValue> values, ulong hash)
    {
        // Dictionary insertion order is not game meaning (CopyFrom can change it).
        ulong sum = 0, xor = 0;
        foreach (var pair in values)
        {
            ulong item = Add(pair.Value, Add(pair.Key, SearchHash.Initial));
            sum = unchecked(sum + item); xor ^= System.Numerics.BitOperations.RotateLeft(item, 23);
        }
        return SearchHash.Add(SearchHash.Add(SearchHash.Add(hash, sum), xor), (ulong)values.Count);
    }
    private static ulong AddSet<T>(ISet<T> values, ulong hash)
    {
        ulong sum = 0, xor = 0;
        foreach (T value in values)
        {
            ulong item = Add(value, SearchHash.Initial);
            sum = unchecked(sum + item); xor ^= System.Numerics.BitOperations.RotateLeft(item, 23);
        }
        return SearchHash.Add(SearchHash.Add(SearchHash.Add(hash, sum), xor), (ulong)values.Count);
    }
    private static readonly ConcurrentDictionary<Type, Func<object, ulong, ulong>> RuntimeReaders = new();
    private static ulong AddRuntime(object value, ulong hash)
    {
        if (value is null) return SearchHash.Add(hash, ulong.MaxValue);
        return RuntimeReaders.GetOrAdd(value.GetType(), type =>
        {
            var input = Expression.Parameter(typeof(object)); var h = Expression.Parameter(typeof(ulong));
            return Expression.Lambda<Func<object, ulong, ulong>>(Expression.Call(typeof(SearchRuleKey), nameof(Add),
                [type], Expression.Convert(input, type), h), input, h).Compile();
        })(value, hash);
    }

    private static bool Skip(Type type, FieldInfo field)
    {
        if (type == typeof(BattleState) && field.Name is "Random" or "<Revision>k__BackingField") return true;
        if (type == typeof(DamageDeckState) && field.Name == "drawPile") return true;
        if (type == typeof(UpgradeCardRuntimeState) && field.Name == "<Definition>k__BackingField") return true;
        if (type == typeof(SearchKnowledge) && field.Name == "<Version>k__BackingField") return true;
        return false;
    }

    private static class Reader<T>
    {
        public static readonly Func<T, ulong, ulong> Hash = Build();
        private static Func<T, ulong, ulong> Build()
        {
            Type type = typeof(T);
            var value = Expression.Parameter(type, "value"); var hash = Expression.Parameter(typeof(ulong), "hash");
            Expression Number(Expression number) => Expression.Call(typeof(SearchHash), nameof(SearchHash.Add), null,
                hash, Expression.Convert(number, typeof(ulong)));
            Expression body;
            if (type == typeof(string))
                body = Expression.Call(typeof(SearchHash), nameof(SearchHash.Add), null, hash, value);
            else if (type == typeof(float))
                body = Number(Expression.Call(typeof(BitConverter), nameof(BitConverter.SingleToUInt32Bits), null, value));
            else if (type == typeof(double))
                body = Number(Expression.Call(typeof(BitConverter), nameof(BitConverter.DoubleToUInt64Bits), null, value));
            else if (type == typeof(bool)) body = Number(Expression.Condition(value, Expression.Constant(1), Expression.Constant(0)));
            else if (type.IsPrimitive || type.IsEnum) body = Number(value);
            else if (type == typeof(DamageCardState))
                body = Expression.Condition(Expression.Property(value, nameof(DamageCardState.FaceUp)),
                    Expression.Call(typeof(SearchRuleKey), nameof(Add), [typeof(CriticalDamageCard?)],
                        Expression.Property(value, nameof(DamageCardState.CriticalCard)),
                        Expression.Call(typeof(SearchHash), nameof(SearchHash.Add), null, hash, Expression.Constant(1UL))),
                    Number(Expression.Constant(0)));
            else if (Nullable.GetUnderlyingType(type) is { } underlying)
                body = Expression.Condition(Expression.Property(value, "HasValue"),
                    Expression.Call(typeof(SearchRuleKey), nameof(Add), [underlying], Expression.Property(value, "Value"),
                        Expression.Call(typeof(SearchHash), nameof(SearchHash.Add), null, hash, Expression.Constant(1UL))),
                    Number(Expression.Constant(0)));
            else if (type == typeof(object) || type.IsInterface || type.IsAbstract)
                body = Expression.Call(typeof(SearchRuleKey), nameof(AddRuntime), null, Expression.Convert(value, typeof(object)), hash);
            else
            {
                var interfaces = type.GetInterfaces().Append(type).ToArray();
                var dictionary = interfaces.FirstOrDefault(t => t.IsGenericType && t.GetGenericTypeDefinition() == typeof(IDictionary<,>));
                var set = interfaces.FirstOrDefault(t => t.IsGenericType && t.GetGenericTypeDefinition() == typeof(ISet<>));
                var sequence = interfaces.FirstOrDefault(t => t.IsGenericType && t.GetGenericTypeDefinition() == typeof(IEnumerable<>));
                if (dictionary is not null)
                    body = Expression.Call(typeof(SearchRuleKey), nameof(AddDictionary), dictionary.GetGenericArguments(), value, hash);
                else if (set is not null)
                    body = Expression.Call(typeof(SearchRuleKey), nameof(AddSet), set.GetGenericArguments(), value, hash);
                else if (sequence is not null)
                    body = Expression.Call(typeof(SearchRuleKey), nameof(AddSequence), sequence.GetGenericArguments(), value, hash);
                else
                {
                    var expressions = new List<Expression>();
                    var accumulator = Expression.Variable(typeof(ulong), "result");
                    expressions.Add(Expression.Assign(accumulator, Expression.Call(typeof(SearchHash), nameof(SearchHash.Add), null,
                        hash, Expression.Constant(SearchHash.Add(SearchHash.Initial, type.FullName ?? type.Name)))));
                    if (type == typeof(DamageDeckState))
                        expressions.Add(Expression.Assign(accumulator, Expression.Call(typeof(SearchHash), nameof(SearchHash.Add), null,
                            accumulator, Expression.Convert(Expression.Property(value, "Remaining"), typeof(ulong)))));
                    foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                        .Where(field => !Skip(type, field)).OrderBy(field => field.Name, StringComparer.Ordinal))
                        expressions.Add(Expression.Assign(accumulator, Expression.Call(typeof(SearchRuleKey), nameof(Add),
                            [field.FieldType], Expression.Field(value, field), accumulator)));
                    expressions.Add(accumulator);
                    body = Expression.Block([accumulator], expressions);
                }
            }
            if (!type.IsValueType)
                body = Expression.Condition(Expression.Equal(value, Expression.Constant(null, type)),
                    Expression.Call(typeof(SearchHash), nameof(SearchHash.Add), null, hash, Expression.Constant(ulong.MaxValue)), body);
            return Expression.Lambda<Func<T, ulong, ulong>>(body, value, hash).Compile();
        }
    }
}
