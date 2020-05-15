
for i in ./fake_test/*.tar.gz; do
    [ -f "$i" ] || break
    tar -xvf "$i" -C ../deepfake_in_the_wild/fake_test
done
