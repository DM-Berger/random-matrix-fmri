#!/bin/bash
TEMPFILE="/tmp/all_found_images.txt"
fd --hidden --no-ignore --regex ".*$1.*.png" | sort > $TEMPFILE;
num_pics=$(< $TEMPFILE wc -l);
thumb_height=500;
thumb_width=960;
montage_width=3840;
per_row=$(($montage_width / $thumb_width));
nrows=$(($num_pics / $per_row));
is_dangling=$(($num_pics % $per_row > 0));
if [ "$is_dangling" -eq 1 ]; then
    nrows=$(($nrows + 1))
fi
montage_height=$(($nrows * thumb_height + 500));
echo "Displaying plots for regex \".*$1.*.png\":"
feh -m \
    --borderless \
    --stretch \
    --thumb-height $thumb_height \
    --thumb-width $thumb_width \
    -H $montage_height \
    -W $montage_width \
    --image-bg white \
    -f $TEMPFILE \
    --title "regex == $1" \
    --thumb-title "regex == $1" \
    ;
rm -rf $TEMPFILE;

