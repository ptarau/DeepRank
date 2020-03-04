rm -r -f "$HOME/Desktop/snapshot"
rm -r -f "$HOME/Desktop/snapshot.tgz"
rsync -av --exclude=".*" \
  --exclude="stanford-corenlp-full-2018-10-05/*" \
  --exclude="dataset/Krapivin2009/*" \
  --exclude="__pycache__" \
    "." "$HOME/Desktop/snapshot/"
pushd .
cd "$HOME/Desktop"
tar zcvf snapshot.tgz snapshot/
rm -r -f "$HOME/Desktop/snapshot"
popd