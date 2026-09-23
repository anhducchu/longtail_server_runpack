# Active vendor policy

`CCUA-DDPM/` is the only active model/data/sampling runtime. Its released DDPM,
CBDM, T2H, and CCUA branches are the complete supported objective set. Active
configs must not import archived research methods or a second model host.

The shared metric cache is `CCUA-DDPM/stats/`; the shared evaluator is
`../tools/evaluate_ccua.py`. Legacy vendor copies were removed; only this
pinned CCUA-DDPM tree may be imported by an active config.

Pinned source provenance is recorded in `THIRD_PARTY_MANIFEST.json`.
