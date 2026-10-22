# Third-party notices for optional stages

MerXen's own code is licensed under the MIT License in `LICENSE`. Optional
stage environments and containers include separately licensed software; the MIT
license does not replace those licenses. No endorsement by Princeton University,
GASTON's contributors, or other dependency authors is implied.

## pciSeq

The optional pciSeq runtime imports the unmodified `pciSeq_3d` package from
https://github.com/acycliq/pciSeq_3d, commit
`5e2c0fd1841f8a4022a2fa341ece965687e343dd`. Its MIT license and copyright
notice are reproduced in `licenses/pciSeq-MIT.txt`. Installed dependencies
retain their own licenses and notices.

## GASTON

- Upstream: https://github.com/raphael-group/GASTON
- Version: gaston-spatial 0.0.2, commit
  `79577c45111b3442b808c8e620711b822965493b`.
- Copyright (c) 2023, Princeton University. All rights reserved.
- License: BSD 3-Clause, reproduced verbatim in
  `licenses/GASTON-BSD-3-Clause.txt`.

MerXen imports the separately installed, unmodified GASTON package. Keep its
copyright notice, conditions and disclaimer with source redistributions and in
the documentation or other materials accompanying binary/container copies.
Do not use the copyright holder's or contributors' names to endorse or promote
derived products without their specific prior written permission.

## GLM-PCA

The optional stage uses Will Townes' unmodified `glmpca` 0.1.0 package:
https://github.com/willtownes/glmpca-py. Its published package metadata specifies
LGPL version 3 or later. The package's license is preserved in
`licenses/glmpca-LICENSE.txt`; `licenses/LGPL-3.0.txt` and
`licenses/GPL-3.0.txt` supply both applicable license texts.

The exact upstream source distribution is included as
`licenses/sources/glmpca-0.1.0.tar.gz`, SHA-256
`02267dc81b1417235d95152a2395d928bff3b503edc5b8d364ebbd1045575118`.
It is unmodified; its upstream release URL is
https://pypi.org/project/glmpca/0.1.0/#files.

GLM-PCA remains a separate Python library loaded at runtime. Users may modify
or replace it with an interface-compatible version and may debug those
modifications, including by reverse engineering the combined work. MerXen
imposes no additional restrictions on those rights. To replace it in a Conda
environment, install your modified library into that environment. For a
container, rebuild `containers/Dockerfile.gaston` with your modified library or
bind-mount a directory containing a modified `glmpca` package and prepend that
directory to `PYTHONPATH`. MerXen's application source and container build recipe
are supplied so the environment can be rebuilt. If redistributing modifications
to GLM-PCA, retain its notices, identify the modifications and dates, and provide
the corresponding modified source under its applicable license.

## Kneed and the remaining environment

Kneed 0.8.5 is BSD 3-Clause, copyright (c) 2017 Kevin Arvai. Its complete notice
is in `licenses/kneed-BSD-3-Clause.txt`.

The GASTON container keeps upstream installed packages and their notices, and
collects Python package license/notice files, metadata and a version inventory
under `/usr/share/merxen/licenses/packages/`. The accompanying MerXen, GASTON,
Kneed, and GLM-PCA notices and GLM-PCA source are under
`/usr/share/merxen/licenses/`. A build fails if the three GASTON-specific
packages or their installed license notices are missing. For a Conda-only
installation, keep each installed package's notices and distribute this notice
and `licenses/` with any bundled environment.

NVIDIA CUDA/cuDNN and operating-system components retain their own licenses
and notices in the base image; these are not relicensed as MIT or BSD. Preserve
them when redistributing the image. The Python inventory is an inventory of the
installed distribution, not a statement that every component has the same
license or a complete source offer for every transitive dependency. Before
redistributing a changed environment, review any new dependencies and their
source, notice and redistribution requirements.
