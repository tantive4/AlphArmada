#nullable disable
namespace Armada.Rules;

public readonly struct WeaponDieProfile
{
	public AttackDieColor Color { get; }

	public string WeaponClass { get; }

	public RangeBand MaximumRange { get; }

	public int MaximumDamage { get; }

	public float BlankChance { get; }

	public float HitIconChance { get; }

	public float CriticalChance { get; }

	public float AccuracyChance { get; }

	public float DoubleHitChance { get; }

	public float HitCriticalChance { get; }

	public float ShipZeroDamageChance => BlankChance + AccuracyChance;

	public float ExpectedDamage { get; }

	public WeaponDieProfile(AttackDieColor color, string weaponClass, RangeBand maximumRange, int maximumDamage, float blankChance, float hitIconChance, float criticalChance, float accuracyChance, float doubleHitChance, float hitCriticalChance, float expectedDamage)
	{
		Color = color;
		WeaponClass = weaponClass;
		MaximumRange = maximumRange;
		MaximumDamage = maximumDamage;
		BlankChance = blankChance;
		HitIconChance = hitIconChance;
		CriticalChance = criticalChance;
		AccuracyChance = accuracyChance;
		DoubleHitChance = doubleHitChance;
		HitCriticalChance = hitCriticalChance;
		ExpectedDamage = expectedDamage;
	}
}
