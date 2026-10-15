#!/bin/bash
# posttag_local.sh TAG -- D145 post-tag, local half: the tag's tree as a tarball on hyde06 (tmp/d145/posttag/tag_<TAG>.tgz).
# Then on hyde06: bash tmp/d145/posttag/posttag_setup.sh TAG
set -e
TAG=$1; W=/c/dev/smb_d145_docs; S=$(dirname "$0")
git -C $W fetch -q --tags origin
git -C $W rev-parse --verify "$TAG^{commit}"
git -C $W archive --format=tar.gz -o "$S/tag_$TAG.tgz" "$TAG"
scp -q -o BatchMode=yes "$S/tag_$TAG.tgz" chayo@hyde06.dabh.io:/data/relcfd/chayo/physmorph_v2/tmp/d145/posttag/
echo "uploaded tag_$TAG.tgz"
