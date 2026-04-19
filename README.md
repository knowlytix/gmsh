# knowlytix / gmsh

**Private rehearsal of the public source mirror for the Knowlytix GMSH
platform.** Mirrors the three Apache-2.0 packages from the `knowlytix/GMS`
private monorepo: `knowlytix-doc`, `knowlytix-bench`, `knowlytix-harness`.

> ⚠️ This repo is a **work in progress**. It is the private staging ground
> for the eventual public `github.com/knowlytix/gmsh`. Before public flip
> (GMS-158), the following land:
>
> - Public-facing README (GMS-155)
> - CONTRIBUTING / SECURITY / CODE_OF_CONDUCT (GMS-155)
> - `tutorials/` populated from `gmsh_testing_ext/notebooks/` (GMS-200)
> - Notebook execution CI (GMS-203)

## Install (rehearsal)

During the private rehearsal, the `knowlytix-core` wheel isn't on PyPI yet.
Install from the `knowlytix/GMS` GitHub Release:

```bash
# Option 1 — from the RC bundle
curl -L -o /tmp/rc.zip \
  https://github.com/knowlytix/GMS/releases/download/v0.1.0-rc1/gms-internal-v0.1.0-rc1.zip
unzip /tmp/rc.zip -d /tmp/rc
pip install /tmp/rc/knowlytix_core-*.whl
pip install -e .[dev]   # doc + bench + harness editable from this clone
```

## License

Apache-2.0 across the repo (source + tutorials). See `LICENSE`.
`knowlytix-core` is a separate commercial package with its own EULA —
see `packaging/knowlytix-core/LICENSE.txt` on PyPI post-publication.
