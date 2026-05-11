# Current Evidence Tables 2026-05-11

This directory is a compact paper-table planning artifact. It collects short
summary values from canonical artifacts after the final-candidate `SC-INR`
seed2/seed3 benchmark was completed.

## Files

- `current_evidence_table.csv`: compact rows for candidate paper tables.
- `provenance.json`: source files used to build the table.

## Intended Use

Use this as a table-planning checklist before editing LaTeX. It keeps the final
`SC-INR` 3-seed PSNR evidence focused on LIIF/LTE comparisons, while leaving
SC-INR vs SC-INR-NoPhi as seed1 context.

## Key Caveats

- Final `SC-INR` is cite-ready for 3-seed benchmark PSNR against LIIF/LTE.
- SC-INR vs `SC-INR-NoPhi` should be shown only as seed1 context, not as a
  seed2/seed3 or main 3-seed comparison.
- Final `SC-INR` auxiliary metrics and qualitative examples are still seed1-only.
- NoSinc consistency is diagnostic, not proof of better observation modeling.
- Selected qualitative figures remain selected examples only.
