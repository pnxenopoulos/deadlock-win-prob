import { useEffect, useState, type ReactNode } from "react";
export function Field({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit,
  immediate = false,
}: {
  label: string;
  value: unknown;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  immediate?: boolean;
}) {
  const [draft, setDraft] = useState(
    typeof value === "number" ? String(Math.round(value * 100) / 100) : "",
  );
  useEffect(
    () =>
      setDraft(
        typeof value === "number" ? String(Math.round(value * 100) / 100) : "",
      ),
    [value],
  );
  const commit = () => {
    const n = draft === "" ? NaN : Number(draft);
    const result = Number.isFinite(n)
      ? Math.min(max ?? Infinity, Math.max(min ?? -Infinity, n))
      : null;
    setDraft(result === null ? "" : String(result));
    onChange(result);
  };
  return (
    <Field label={label}>
      <span className="input-unit">
        <input
          type="number"
          aria-label={label}
          value={draft}
          min={min}
          max={max}
          step={step}
          placeholder="Unknown"
          onChange={(e) => {
            setDraft(e.target.value);
            const n = e.target.valueAsNumber;
            if (immediate && Number.isFinite(n))
              onChange(
                Math.min(max ?? Infinity, Math.max(min ?? -Infinity, n)),
              );
          }}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
          }}
        />
        {unit && <span>{unit}</span>}
      </span>
    </Field>
  );
}
export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="check-field">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}
