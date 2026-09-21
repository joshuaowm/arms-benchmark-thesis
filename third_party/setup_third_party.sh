#!/usr/bin/env bash
# Clone the third-party model repos into code/ in the project folder (the folder that holds
# this repository, $BASE in the other scripts) at the commits used for the thesis, and apply
# our two changes:
#   OSISeg      dataset/utils.py: np.int -> np.int64 (np.int was removed from numpy), made
#               here with sed because OSISeg has no license and its code is not copied here
#   SimpleClick isegm/data/transforms.py: albumentations import path for albumentations 1.x,
#               copied from overlay/ (MIT, see overlay/SimpleClick/LICENSE; diff in diffs/)
#   SegNext     unmodified
#   sam-hq      unmodified (SAM2 is inside it, under sam-hq2/)
# Then run scripts/install_specialist_ft.sh to add the SegNext / SimpleClick fine-tune configs.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO/.."
mkdir -p code
clone() { [ -d "code/$1/.git" ] || git clone "$2" "code/$1"; git -C "code/$1" checkout -q "$3"; }
clone OSISeg https://github.com/zhilyzhang/OSISeg.git fcfac0ebadfd1f98887b1d4b8d1dc741dbac12ef
clone sam-hq https://github.com/SysCV/sam-hq.git e696978d60352dc9a26b12631cd91781502c6546
clone SegNext https://github.com/uncbiag/SegNext.git 4c45ce8bfa8d3121d36d71f0ff263555805dad89
clone SimpleClick https://github.com/uncbiag/SimpleClick.git 2f4efd2d9800732b7d509ac4343756e628bc91fe
sed -i 's/dtype=np\.int)/dtype=np.int64)/' code/OSISeg/dataset/utils.py
grep -q 'dtype=np.int64)' code/OSISeg/dataset/utils.py
cp -r "$REPO/third_party/overlay/." code/
echo "done: $(pwd)/code. Model weights are not in git; see the README."
