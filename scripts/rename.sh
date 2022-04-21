ls -AbQR1p * |                        # paths in "quoted" \escaped format
    sed -n '/":$/,${/.[^/]$/p}' |     # skip files in pwd, blank and dir/ lines
    { bwd="$PWD"; while read -n1;     # save base dir strip leading "quote
                        read -r p; do # read path
        printf -vs "${p%\"*}"         # strip trailing quote"(:)
        [[ $p == *: ]] && {           # this is a directory
            cd   "$bwd/$s"            # change into new directory
            rnp="${s//\//-}"-         # relative name prefix
        } || mv "$s" "$bwd/$rnp$s"
      done; }
