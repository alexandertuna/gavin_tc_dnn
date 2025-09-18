for a in 0.25 0.5 1.0 2.0 4.0; do
  for b in 0.25 0.5 1.0 2.0 4.0; do
    for c in 0.25 0.5 1.0 2.0 4.0; do
      echo "$a $b $c" >> grid.txt
    done
  done
done
